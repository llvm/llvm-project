//===- LowerToLLVMOffloadPrintfAMDGPU.cpp - AMDGPU printf lowering --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers cir.offload.printf to the hostcall and buffered printf
// runtime interfaces of AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVM.h"
#include "LowerToLLVMOffloadPrintf.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/AMDGPUPrintf.h"
#include "llvm/Support/MathExtras.h"

namespace cir {
namespace direct {

namespace {

std::optional<int64_t> getConstantInt(mlir::Value value) {
  auto constOp = value.getDefiningOp<cir::ConstantOp>();
  auto intAttr =
      constOp ? mlir::dyn_cast<cir::IntAttr>(constOp.getValue()) : nullptr;
  if (!intAttr)
    return std::nullopt;
  const llvm::APInt &v = intAttr.getValue();
  if (mlir::cast<cir::IntType>(intAttr.getType()).isSigned()
          ? v.getSignificantBits() > 64
          : v.getActiveBits() > 63)
    return std::nullopt;
  return v.getSExtValue();
}

bool isByteType(mlir::Type ty) {
  auto intTy = mlir::dyn_cast<cir::IntType>(ty);
  return intTy && intTy.getWidth() == 8;
}

bool getConstantStringInfo(mlir::Value ptr, mlir::ModuleOp mod,
                           mlir::SymbolTableCollection &symbolTables,
                           std::string &str) {
  int64_t offset = 0;
  while (mlir::Operation *def = ptr.getDefiningOp()) {
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() != cir::CastKind::array_to_ptrdecay &&
          cast.getKind() != cir::CastKind::bitcast &&
          cast.getKind() != cir::CastKind::address_space)
        break;
      ptr = cast.getSrc();
    } else if (auto elt = mlir::dyn_cast<cir::GetElementOp>(def)) {
      std::optional<int64_t> index = getConstantInt(elt.getIndex());
      if (!index || !isByteType(elt.getElementType()) ||
          llvm::AddOverflow(offset, *index, offset))
        return false;
      ptr = elt.getBase();
    } else if (auto stride = mlir::dyn_cast<cir::PtrStrideOp>(def)) {
      std::optional<int64_t> index = getConstantInt(stride.getStride());
      if (!index || !isByteType(stride.getElementType()) ||
          llvm::AddOverflow(offset, *index, offset))
        return false;
      ptr = stride.getBase();
    } else {
      break;
    }
  }

  auto getGlobal = ptr.getDefiningOp<cir::GetGlobalOp>();
  if (!getGlobal)
    return false;

  // The global may or may not have been converted already.
  std::string data;
  mlir::Operation *symbol =
      symbolTables.lookupSymbolIn(mod, getGlobal.getNameAttr());
  if (auto global = mlir::dyn_cast_if_present<cir::GlobalOp>(symbol)) {
    std::optional<mlir::Attribute> init = global.getInitialValue();
    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(global.getSymType());
    if (!global.getConstant() || !init || !arrayTy ||
        !isByteType(arrayTy.getElementType()))
      return false;
    if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(*init)) {
      auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constArr.getElts());
      if (!strAttr)
        return false;
      data = strAttr.getValue().str();
    } else if (!mlir::isa<cir::ZeroAttr>(*init)) {
      return false;
    }
    // Add back the trailing NULs the string attribute leaves out.
    data.resize(arrayTy.getSize(), '\0');
  } else if (auto global =
                 mlir::dyn_cast_if_present<mlir::LLVM::GlobalOp>(symbol)) {
    if (!global.getConstant())
      return false;
    mlir::Attribute value = global.getValueOrNull();
    if (auto strAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(value)) {
      data = strAttr.getValue().str();
    } else if (auto dense =
                   mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(value);
               dense && dense.getElementType().isInteger(8)) {
      for (const llvm::APInt &c : dense.getValues<llvm::APInt>())
        data.push_back(static_cast<char>(c.getZExtValue()));
    } else {
      return false;
    }
  } else {
    return false;
  }

  if (offset < 0 || static_cast<uint64_t>(offset) > data.size())
    return false;
  llvm::StringRef rest = llvm::StringRef(data).drop_front(offset);
  str = rest.substr(0, rest.find('\0')).str();
  return true;
}

// helper struct to package the string related data
struct StringData {
  std::string str;
  mlir::Value realSize;
  mlir::Value alignedSize;
  bool isConst = true;
};

class AMDGPUPrintfLowering {
public:
  AMDGPUPrintfLowering(cir::OffloadPrintfOp op,
                       mlir::ConversionPatternRewriter &rewriter,
                       const mlir::DataLayout &dataLayout,
                       mlir::SymbolTableCollection &symbolTables,
                       unsigned genericAddrSpace)
      : op(op), rewriter(rewriter), dataLayout(dataLayout),
        symbolTables(symbolTables), mod(op->getParentOfType<mlir::ModuleOp>()),
        genericAddrSpace(genericAddrSpace) {}

  mlir::LogicalResult lower(mlir::ValueRange args, bool isBuffered);

private:
  mlir::Value getI32(uint32_t value) {
    return mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), llvm::APInt(32, value));
  }
  mlir::Value getI64(uint64_t value) {
    return mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), llvm::APInt(64, value));
  }
  mlir::Value createCall(llvm::StringRef name, mlir::Type resTy,
                         mlir::ValueRange args);
  mlir::Value gepInBounds(mlir::Value ptr, mlir::LLVM::GEPArg offset);
  void addPrintfFormatMetadata(llvm::StringRef entry, bool onlyIfEmpty);

  mlir::Value fitArgInto64Bits(mlir::Value arg);
  mlir::Value callPrintfBegin(mlir::Value version);
  mlir::Value callAppendArgs(mlir::Value desc, int numArgs, mlir::Value arg0,
                             mlir::Value arg1, mlir::Value arg2,
                             mlir::Value arg3, mlir::Value arg4,
                             mlir::Value arg5, mlir::Value arg6, bool isLast);
  mlir::Value appendArg(mlir::Value desc, mlir::Value arg, bool isLast);
  mlir::Value getStrlenWithNull(mlir::Value str);
  mlir::Value callAppendStringN(mlir::Value desc, mlir::Value str,
                                mlir::Value length, bool isLast);
  mlir::Value appendString(mlir::Value desc, mlir::Value arg, bool isLast);
  mlir::Value processArg(mlir::Value desc, mlir::Value arg, bool specIsCString,
                         bool isLast);
  bool isCStringArg(mlir::ValueRange args, size_t i) const;
  mlir::Value callBufferedPrintfStart(
      mlir::ValueRange args, mlir::Value fmt, bool isConstFmtStr,
      llvm::SmallVectorImpl<StringData> &stringContents, mlir::Value &argSize);
  void
  processConstantStringArg(const StringData &sd,
                           llvm::SmallVectorImpl<mlir::Value> &whatToStore);
  mlir::Value processNonStringArg(mlir::Value arg);
  void callBufferedPrintfArgPush(mlir::ValueRange args, mlir::Value ptrToStore,
                                 llvm::ArrayRef<StringData> stringContents,
                                 bool isConstFmtStr);
  mlir::Value emitAMDGPUPrintfCall(mlir::ValueRange args, bool isBuffered);

  cir::OffloadPrintfOp op;
  mlir::ConversionPatternRewriter &rewriter;
  const mlir::DataLayout &dataLayout;
  mlir::SymbolTableCollection &symbolTables;
  mlir::ModuleOp mod;
  unsigned genericAddrSpace;

  /// The rest of the op's block, which the expansion branches to.
  mlir::Block *cont = nullptr;
  std::string fmtStr;
  llvm::SparseBitVector<8> specIsCString;
};

} // namespace

mlir::Value AMDGPUPrintfLowering::createCall(llvm::StringRef name,
                                             mlir::Type resTy,
                                             mlir::ValueRange args) {
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(
      resTy, llvm::to_vector(args.getTypes()), /*isVarArg=*/false);
  createLLVMFuncOpIfNotExist(rewriter, symbolTables, op, name, fnTy);
  return mlir::LLVM::CallOp::create(
             rewriter, op.getLoc(), fnTy,
             mlir::FlatSymbolRefAttr::get(rewriter.getContext(), name), args)
      .getResult();
}

mlir::Value AMDGPUPrintfLowering::gepInBounds(mlir::Value ptr,
                                              mlir::LLVM::GEPArg offset) {
  return mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptr.getType(),
                                   rewriter.getI8Type(), ptr,
                                   llvm::ArrayRef<mlir::LLVM::GEPArg>{offset},
                                   mlir::LLVM::GEPNoWrapFlags::inbounds);
}

void AMDGPUPrintfLowering::addPrintfFormatMetadata(llvm::StringRef entry,
                                                   bool onlyIfEmpty) {
  constexpr llvm::StringLiteral name = "llvm.printf.fmts";
  mlir::Attribute node = mlir::LLVM::MDNodeAttr::get(
      rewriter.getContext(),
      {mlir::LLVM::MDStringAttr::get(
          rewriter.getContext(),
          mlir::StringAttr::get(rewriter.getContext(), entry))});

  mlir::LLVM::NamedMetadataOp metaD;
  for (auto md : mod.getOps<mlir::LLVM::NamedMetadataOp>()) {
    if (md.getMetadataName() == name) {
      metaD = md;
      break;
    }
  }

  if (!metaD) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(mod.getBody());
    mlir::LLVM::NamedMetadataOp::create(rewriter, mod.getLoc(), name,
                                        rewriter.getArrayAttr(node));
    return;
  }

  if (onlyIfEmpty && !metaD.getNodes().empty())
    return;
  llvm::SmallVector<mlir::Attribute> nodes(metaD.getNodes().getValue());
  nodes.push_back(node);
  rewriter.modifyOpInPlace(
      metaD, [&] { metaD.setNodesAttr(rewriter.getArrayAttr(nodes)); });
}

mlir::Value AMDGPUPrintfLowering::fitArgInto64Bits(mlir::Value arg) {
  mlir::Type int64Ty = rewriter.getI64Type();
  mlir::Type ty = arg.getType();

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    switch (intTy.getWidth()) {
    case 32:
      return mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(), int64Ty, arg);
    case 64:
      return arg;
    }
  }

  if (ty.isF64()) {
    return mlir::LLVM::BitcastOp::create(rewriter, op.getLoc(), int64Ty, arg);
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(ty)) {
    return mlir::LLVM::PtrToIntOp::create(rewriter, op.getLoc(), int64Ty, arg);
  }

  llvm_unreachable("unexpected type");
}

mlir::Value AMDGPUPrintfLowering::callPrintfBegin(mlir::Value version) {
  return createCall("__ockl_printf_begin", rewriter.getI64Type(), version);
}

mlir::Value AMDGPUPrintfLowering::callAppendArgs(
    mlir::Value desc, int numArgs, mlir::Value arg0, mlir::Value arg1,
    mlir::Value arg2, mlir::Value arg3, mlir::Value arg4, mlir::Value arg5,
    mlir::Value arg6, bool isLast) {
  mlir::Value isLastValue = getI32(isLast);
  mlir::Value numArgsValue = getI32(numArgs);
  return createCall("__ockl_printf_append_args", rewriter.getI64Type(),
                    {desc, numArgsValue, arg0, arg1, arg2, arg3, arg4, arg5,
                     arg6, isLastValue});
}

mlir::Value AMDGPUPrintfLowering::appendArg(mlir::Value desc, mlir::Value arg,
                                            bool isLast) {
  mlir::Value arg0 = fitArgInto64Bits(arg);
  mlir::Value zero = getI64(0);
  return callAppendArgs(desc, 1, arg0, zero, zero, zero, zero, zero, zero,
                        isLast);
}

// The device library does not provide strlen, so we build our own loop
// here. While we are at it, we also include the terminating null in the length.
mlir::Value AMDGPUPrintfLowering::getStrlenWithNull(mlir::Value str) {
  mlir::Block *prev = rewriter.getInsertionBlock();
  mlir::Region *region = cont->getParent();

  mlir::Value charZero = mlir::LLVM::ConstantOp::create(
      rewriter, op.getLoc(), rewriter.getI8Type(), llvm::APInt(8, 0));
  mlir::Value one = getI64(1);
  mlir::Value zero = getI64(0);
  mlir::Type int64Ty = rewriter.getI64Type();

  // The length is either zero for a null pointer, or the computed value for an
  // actual string. We need a join block for a phi that represents the final
  // value.
  //
  //  Strictly speaking, the zero does not matter since
  // __ockl_printf_append_string_n ignores the length if the pointer is null.
  mlir::Block *join = rewriter.createBlock(region, cont->getIterator(),
                                           {int64Ty}, {op.getLoc()});
  mlir::Block *whileBlock = rewriter.createBlock(
      region, join->getIterator(), {str.getType()}, {op.getLoc()});
  mlir::Block *whileDone = rewriter.createBlock(region, join->getIterator());

  // Emit an early return for when the pointer is null.
  rewriter.setInsertionPointToEnd(prev);
  mlir::Value cmpNull = mlir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, str,
      mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), str.getType()));
  mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmpNull, join, zero,
                               whileBlock, str);

  // Entry to the while loop.
  rewriter.setInsertionPointToEnd(whileBlock);

  mlir::Value ptrPhi = whileBlock->getArgument(0);
  mlir::Value ptrNext = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), str.getType(), rewriter.getI8Type(), ptrPhi,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{one});

  // Condition for the while loop.
  mlir::Value data = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                                rewriter.getI8Type(), ptrPhi);
  mlir::Value cmp = mlir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, data, charZero);
  mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmp, whileDone,
                               mlir::ValueRange(), whileBlock, ptrNext);

  // Add one to the computed length.
  rewriter.setInsertionPointToEnd(whileDone);
  auto addrTy = mlir::IntegerType::get(
      rewriter.getContext(), *dataLayout.getTypeIndexBitwidth(str.getType()));
  mlir::Value endAddr =
      mlir::LLVM::PtrToAddrOp::create(rewriter, op.getLoc(), addrTy, ptrPhi);
  mlir::Value beginAddr =
      mlir::LLVM::PtrToAddrOp::create(rewriter, op.getLoc(), addrTy, str);
  mlir::Value len =
      mlir::LLVM::SubOp::create(rewriter, op.getLoc(), endAddr, beginAddr);
  if (addrTy != int64Ty)
    len = mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(), int64Ty, len);
  len = mlir::LLVM::AddOp::create(rewriter, op.getLoc(), len, one);

  // Final join.
  mlir::LLVM::BrOp::create(rewriter, op.getLoc(), len, join);
  rewriter.setInsertionPointToEnd(join);
  return join->getArgument(0);
}

mlir::Value AMDGPUPrintfLowering::callAppendStringN(mlir::Value desc,
                                                    mlir::Value str,
                                                    mlir::Value length,
                                                    bool isLast) {
  mlir::Value isLastInt32 = getI32(isLast);
  // __ockl_printf_append_string_n takes a generic pointer, which can address a
  // string in any address space.
  auto ptrTy =
      mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), genericAddrSpace);
  if (str.getType() != ptrTy)
    str =
        mlir::LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), ptrTy, str);
  return createCall("__ockl_printf_append_string_n", rewriter.getI64Type(),
                    {desc, str, length, isLastInt32});
}

mlir::Value AMDGPUPrintfLowering::appendString(mlir::Value desc,
                                               mlir::Value arg, bool isLast) {
  mlir::Value length = getStrlenWithNull(arg);
  return callAppendStringN(desc, arg, length, isLast);
}

mlir::Value AMDGPUPrintfLowering::processArg(mlir::Value desc, mlir::Value arg,
                                             bool specIsCString, bool isLast) {
  if (specIsCString && mlir::isa<mlir::LLVM::LLVMPointerType>(arg.getType())) {
    return appendString(desc, arg, isLast);
  }
  // If the format specifies a string but the argument is not, the frontend will
  // have printed a warning. We just rely on undefined behaviour and send the
  // argument anyway.
  return appendArg(desc, arg, isLast);
}

// As in processArg, a non-pointer argument for a string is sent as a scalar.
bool AMDGPUPrintfLowering::isCStringArg(mlir::ValueRange args, size_t i) const {
  return specIsCString.test(i) &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(args[i].getType());
}

// Calculates frame size required for current printf expansion and allocates
// space on printf buffer. Printf frame includes following contents
// [ ControlDWord , format string/Hash , Arguments (each aligned to 8 byte) ]
mlir::Value AMDGPUPrintfLowering::callBufferedPrintfStart(
    mlir::ValueRange args, mlir::Value fmt, bool isConstFmtStr,
    llvm::SmallVectorImpl<StringData> &stringContents, mlir::Value &argSize) {
  mlir::Value nonConstStrLen;
  mlir::Value lenWithNull;
  mlir::Value lenWithNullAligned;
  mlir::Value tempAdd;

  // First 4 bytes to be reserved for control dword
  size_t bufSize = 4;
  if (isConstFmtStr)
    // First 8 bytes of MD5 hash
    bufSize += 8;
  else {
    lenWithNull = getStrlenWithNull(fmt);

    // Align the computed length to next 8 byte boundary
    tempAdd = mlir::LLVM::AddOp::create(rewriter, op.getLoc(), lenWithNull,
                                        getI64(7U));
    nonConstStrLen =
        mlir::LLVM::AndOp::create(rewriter, op.getLoc(), tempAdd, getI64(~7U));

    stringContents.push_back({"", lenWithNull, nonConstStrLen, false});
  }

  mlir::OperandRange cirArgs = op->getOperands();
  for (size_t i = 1; i < args.size(); i++) {
    if (isCStringArg(args, i)) {
      std::string argStr;
      if (getConstantStringInfo(cirArgs[i], mod, symbolTables, argStr)) {
        auto alignedLen = llvm::alignTo(argStr.size() + 1, 8);
        stringContents.push_back({argStr, /*realSize=*/{}, /*alignedSize=*/{},
                                  /*isConst=*/true});
        bufSize += alignedLen;
      } else {
        lenWithNull = getStrlenWithNull(args[i]);

        // Align the computed length to next 8 byte boundary
        tempAdd = mlir::LLVM::AddOp::create(rewriter, op.getLoc(), lenWithNull,
                                            getI64(7U));
        lenWithNullAligned = mlir::LLVM::AndOp::create(rewriter, op.getLoc(),
                                                       tempAdd, getI64(~7U));

        if (nonConstStrLen) {
          mlir::Value val = mlir::LLVM::AddOp::create(
              rewriter, op.getLoc(), lenWithNullAligned, nonConstStrLen);
          nonConstStrLen = val;
        } else
          nonConstStrLen = lenWithNullAligned;

        stringContents.push_back({"", lenWithNull, lenWithNullAligned, false});
      }
    } else {
      uint64_t allocSize =
          dataLayout.getTypeAllocSize(args[i].getType()).getFixedValue();
      // We end up expanding non string arguments to 8 bytes
      // (args smaller than 8 bytes)
      bufSize += llvm::alignTo(allocSize, 8);
    }
  }

  // calculate final size value to be passed to printf_alloc
  mlir::Value sizeToReserve = getI64(bufSize);
  if (nonConstStrLen)
    sizeToReserve = mlir::LLVM::AddOp::create(rewriter, op.getLoc(),
                                              nonConstStrLen, sizeToReserve);

  argSize = mlir::LLVM::TruncOp::create(rewriter, op.getLoc(),
                                        rewriter.getI32Type(), sizeToReserve);

  // call the printf_alloc function
  unsigned globalAS = 0;
  if (auto as = mlir::dyn_cast_if_present<mlir::IntegerAttr>(
          dataLayout.getGlobalMemorySpace()))
    globalAS = as.getValue().getZExtValue();
  auto ptrTy =
      mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), globalAS);
  auto allocName =
      mlir::StringAttr::get(rewriter.getContext(), "__printf_alloc");
  bool declared = symbolTables.lookupSymbolIn(mod, allocName);
  mlir::Value ptr = createCall(allocName, ptrTy, argSize);
  if (!declared)
    mlir::cast<mlir::LLVM::LLVMFuncOp>(
        symbolTables.lookupSymbolIn(mod, allocName))
        .setNoUnwind(true);
  return ptr;
}

// Prepare constant string argument to push onto the buffer
void AMDGPUPrintfLowering::processConstantStringArg(
    const StringData &sd, llvm::SmallVectorImpl<mlir::Value> &whatToStore) {
  llvm::SmallVector<uint32_t, 16> words;
  llvm::packAMDGPUPrintfConstantString(sd.str, words);
  for (uint32_t word : words)
    whatToStore.push_back(getI32(word));
}

mlir::Value AMDGPUPrintfLowering::processNonStringArg(mlir::Value arg) {
  mlir::Type ty = arg.getType();

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    if (intTy.getWidth() < 64) {
      return mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(),
                                        rewriter.getI64Type(), arg);
    }
  }

  if (mlir::isa<mlir::FloatType>(ty)) {
    if (dataLayout.getTypeAllocSize(ty).getFixedValue() < 8) {
      return mlir::LLVM::FPExtOp::create(rewriter, op.getLoc(),
                                         rewriter.getF64Type(), arg);
    }
  }

  return arg;
}

void AMDGPUPrintfLowering::callBufferedPrintfArgPush(
    mlir::ValueRange args, mlir::Value ptrToStore,
    llvm::ArrayRef<StringData> stringContents, bool isConstFmtStr) {
  const StringData *strIt = stringContents.begin();
  size_t i = isConstFmtStr ? 1 : 0;
  for (; i < args.size(); i++) {
    if (i != 0 && !isCStringArg(args, i)) {
      // The runtime reads each non-string argument from a slot of at least 8
      // bytes, also when the stored value is smaller.
      mlir::Value toStore = processNonStringArg(args[i]);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), toStore, ptrToStore);
      uint64_t allocSize =
          dataLayout.getTypeAllocSize(toStore.getType()).getFixedValue();
      ptrToStore = gepInBounds(ptrToStore, llvm::alignTo(allocSize, 8));
      continue;
    }

    if (!strIt->isConst) {
      // This copies the contents of the string, however the next offset
      // is at aligned length, the extra space that might be created due
      // to alignment padding is not populated with any specific value
      // here. This would be safe as long as runtime is sync with
      // the offsets.
      auto memcpy = mlir::LLVM::MemcpyOp::create(
          rewriter, op.getLoc(), ptrToStore, args[i], strIt->realSize,
          /*isVolatile=*/false);
      mlir::NamedAttribute align1(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                                  rewriter.getI64IntegerAttr(1));
      memcpy.setArgAttrsAttr(
          rewriter.getArrayAttr({rewriter.getDictionaryAttr(align1),
                                 rewriter.getDictionaryAttr(align1),
                                 rewriter.getDictionaryAttr({})}));

      ptrToStore = gepInBounds(ptrToStore, strIt->alignedSize);

      // done with current argument, move to next
      strIt++;
      continue;
    }

    llvm::SmallVector<mlir::Value, 32> whatToStore;
    processConstantStringArg(*strIt, whatToStore);
    strIt++;

    for (mlir::Value toStore : whatToStore) {
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), toStore, ptrToStore);
      ptrToStore = gepInBounds(
          ptrToStore,
          dataLayout.getTypeAllocSize(toStore.getType()).getFixedValue());
    }
  }
}

mlir::Value AMDGPUPrintfLowering::emitAMDGPUPrintfCall(mlir::ValueRange args,
                                                       bool isBuffered) {
  size_t numOps = args.size();
  assert(numOps >= 1);

  mlir::Value fmt = args[0];

  if (getConstantStringInfo(op.getFormat(), mod, symbolTables, fmtStr))
    llvm::locateAMDGPUPrintfCStrings(specIsCString, fmtStr);

  if (isBuffered) {
    llvm::SmallVector<StringData, 8> stringContents;
    bool isConstFmtStr = !fmtStr.empty();

    mlir::Value argSize;
    mlir::Value ptr = callBufferedPrintfStart(args, fmt, isConstFmtStr,
                                              stringContents, argSize);

    // The buffered version still follows OpenCL printf standards for
    // printf return value, i.e 0 on success, -1 on failure.
    mlir::Value cmp = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::ne, ptr,
        mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptr.getType()));

    // The rest of the op's block is the end block.
    mlir::Block *end = cont;
    mlir::Block *argPush =
        rewriter.createBlock(cont->getParent(), cont->getIterator());

    rewriter.setInsertionPointToEnd(cmp.getParentBlock());
    mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmp, argPush, end);
    rewriter.setInsertionPointToEnd(argPush);

    // Create controlDWord and store as the first entry, format as follows
    // Bit 0 (LSB) -> stream (1 if stderr, 0 if stdout, printf always outputs to
    // stdout) Bit 1 -> constant format string (1 if constant) Bits 2-31 -> size
    // of printf data frame
    mlir::Value constantTwo = getI32(2);
    mlir::Value controlDWord =
        mlir::LLVM::ShlOp::create(rewriter, op.getLoc(), argSize, constantTwo);
    if (isConstFmtStr)
      controlDWord = mlir::LLVM::OrOp::create(rewriter, op.getLoc(),
                                              controlDWord, constantTwo);

    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), controlDWord, ptr);

    ptr = gepInBounds(ptr, 4);

    // Create MD5 hash for constant format string, push low 64 bits of the
    // same onto buffer and metadata.
    if (isConstFmtStr) {
      addPrintfFormatMetadata(llvm::getAMDGPUPrintfFormatMetadata(fmtStr),
                              /*onlyIfEmpty=*/false);

      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(),
          getI64(llvm::getAMDGPUPrintfFormatHash(fmtStr)), ptr);
      ptr = gepInBounds(ptr, 8);
    } else {
      // Include a dummy metadata instance in case of only non constant
      // format string usage, This might be an absurd usecase but needs to
      // be done for completeness
      addPrintfFormatMetadata(llvm::AMDGPUPrintfNonConstFormatMetadata,
                              /*onlyIfEmpty=*/true);
    }

    // Push The printf arguments onto buffer
    callBufferedPrintfArgPush(args, ptr, stringContents, isConstFmtStr);

    // End block, returns -1 on failure
    mlir::LLVM::BrOp::create(rewriter, op.getLoc(), end);
    rewriter.setInsertionPointToStart(end);
    mlir::Value notCmp = mlir::LLVM::XOrOp::create(
        rewriter, op.getLoc(), cmp,
        mlir::LLVM::ConstantOp::create(
            rewriter, op.getLoc(), rewriter.getI1Type(), llvm::APInt(1, 1)));
    return mlir::LLVM::SExtOp::create(rewriter, op.getLoc(),
                                      rewriter.getI32Type(), notCmp);
  }

  mlir::Value desc = callPrintfBegin(getI64(0));
  desc = appendString(desc, fmt, numOps == 1);

  // FIXME: This invokes hostcall once for each argument. We can pack up to
  // seven scalar printf arguments in a single hostcall. See the signature of
  // callAppendArgs().
  for (unsigned int i = 1; i != numOps; ++i) {
    bool isLast = i == numOps - 1;
    bool isCString = specIsCString.test(i);
    desc = processArg(desc, args[i], isCString, isLast);
  }

  return mlir::LLVM::TruncOp::create(rewriter, op.getLoc(),
                                     rewriter.getI32Type(), desc);
}

mlir::LogicalResult AMDGPUPrintfLowering::lower(mlir::ValueRange args,
                                                bool isBuffered) {
  // Unlike fitArgInto64Bits's unreachable, report a type hostcall cannot send.
  if (!isBuffered) {
    for (auto [cirArg, arg] : llvm::zip(op.getArgs(), args.drop_front())) {
      mlir::Type ty = arg.getType();
      if (ty.isInteger(32) || ty.isInteger(64) || ty.isF64() ||
          mlir::isa<mlir::LLVM::LLVMPointerType>(ty))
        continue;
      return op.emitError() << "unsupported argument type " << cirArg.getType()
                            << " for AMDGPU printf";
    }
  }

  mlir::Block *block = op->getBlock();
  cont = rewriter.splitBlock(block, mlir::Block::iterator(op));
  rewriter.setInsertionPointToEnd(block);

  mlir::Value result = emitAMDGPUPrintfCall(args, isBuffered);
  if (rewriter.getInsertionBlock() != cont)
    mlir::LLVM::BrOp::create(rewriter, op.getLoc(), cont);

  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult lowerAMDGPUPrintf(cir::OffloadPrintfOp op,
                                      mlir::ValueRange operands,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      const mlir::DataLayout &dataLayout,
                                      mlir::SymbolTableCollection &symbolTables,
                                      unsigned genericAddrSpace,
                                      bool isBuffered) {
  return AMDGPUPrintfLowering(op, rewriter, dataLayout, symbolTables,
                              genericAddrSpace)
      .lower(operands, isBuffered);
}

} // namespace direct
} // namespace cir
