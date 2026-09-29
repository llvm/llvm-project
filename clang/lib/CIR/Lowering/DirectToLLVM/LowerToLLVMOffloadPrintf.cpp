//===- LowerToLLVMOffloadPrintf.cpp - Offload printf lowering -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

namespace cir {
namespace direct {

namespace {

/// Get the contents of the constant string ptr points to, up to the first NUL.
/// This is the counterpart of llvm::getConstantStringInfo, and looks through
/// the CIR a string literal is addressed with.
bool getConstantStringInfo(mlir::Value ptr, mlir::ModuleOp mod,
                           mlir::SymbolTableCollection &symbolTables,
                           std::string &str) {
  auto cast = ptr.getDefiningOp<cir::CastOp>();
  while (cast && (cast.getKind() == cir::CastKind::array_to_ptrdecay ||
                  cast.getKind() == cir::CastKind::bitcast ||
                  cast.getKind() == cir::CastKind::address_space)) {
    ptr = cast.getSrc();
    cast = ptr.getDefiningOp<cir::CastOp>();
  }

  auto getGlobal = ptr.getDefiningOp<cir::GetGlobalOp>();
  if (!getGlobal)
    return false;

  // The global may or may not have been converted already.
  mlir::Operation *symbol =
      symbolTables.lookupSymbolIn(mod, getGlobal.getNameAttr());
  llvm::StringRef data;
  if (auto global = mlir::dyn_cast_if_present<cir::GlobalOp>(symbol)) {
    std::optional<mlir::Attribute> init = global.getInitialValue();
    if (!global.getConstant() || !init)
      return false;
    if (mlir::isa<cir::ZeroAttr>(*init)) {
      str.clear();
      return true;
    }
    auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(*init);
    auto strAttr = constArr
                       ? mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())
                       : nullptr;
    if (!strAttr)
      return false;
    data = strAttr.getValue();
  } else if (auto global =
                 mlir::dyn_cast_if_present<mlir::LLVM::GlobalOp>(symbol)) {
    if (!global.getConstant())
      return false;
    mlir::Attribute value = global.getValueOrNull();
    if (auto strAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(value)) {
      data = strAttr.getValue();
    } else if (auto dense =
                   mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(value);
               dense && dense.getElementType().isInteger(8)) {
      str.clear();
      for (const llvm::APInt &c : dense.getValues<llvm::APInt>()) {
        if (c.isZero())
          break;
        str.push_back(static_cast<char>(c.getZExtValue()));
      }
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }

  str = data.substr(0, data.find('\0')).str();
  return true;
}

// Helper struct to package the string related data.
struct StringData {
  std::string str;
  mlir::Value realSize;
  mlir::Value alignedSize;
  bool isConst = true;
};

/// Expands one cir.offload.printf into the AMDGPU printf runtime sequence.
///
/// Everything after the op is split off into a continuation block first, so
/// the expansion can build control flow of its own.
class AMDGPUPrintfLowering {
public:
  AMDGPUPrintfLowering(cir::OffloadPrintfOp op,
                       mlir::ConversionPatternRewriter &rewriter,
                       const mlir::DataLayout &dataLayout,
                       mlir::SymbolTableCollection &symbolTables)
      : op(op), rewriter(rewriter), dataLayout(dataLayout),
        symbolTables(symbolTables), mod(op->getParentOfType<mlir::ModuleOp>()) {
  }

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

  mlir::Value fitArgInto64Bits(mlir::Value arg);
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
  mlir::Value emitHostcall(mlir::ValueRange args);

  bool isCStringArg(mlir::ValueRange args, size_t i) const {
    // The format specifies a string but the argument is not a pointer. The
    // frontend will have warned; send the argument as a scalar.
    return specIsCString.test(i) &&
           mlir::isa<mlir::LLVM::LLVMPointerType>(args[i].getType());
  }
  mlir::Value alignTo8(mlir::Value len);
  mlir::Value
  callBufferedPrintfStart(mlir::ValueRange args, bool isConstFmtStr,
                          llvm::SmallVectorImpl<StringData> &stringContents,
                          mlir::Value &argSize);
  void
  processConstantStringArg(const StringData &sd,
                           llvm::SmallVectorImpl<mlir::Value> &whatToStore);
  mlir::Value processNonStringArg(mlir::Value arg);
  void callBufferedPrintfArgPush(mlir::ValueRange args, mlir::Value ptrToStore,
                                 llvm::ArrayRef<StringData> stringContents,
                                 bool isConstFmtStr);
  void addPrintfFormatMetadata(llvm::StringRef entry, bool onlyIfEmpty);
  mlir::Value emitBuffered(mlir::ValueRange args);

  cir::OffloadPrintfOp op;
  mlir::ConversionPatternRewriter &rewriter;
  const mlir::DataLayout &dataLayout;
  mlir::SymbolTableCollection &symbolTables;
  mlir::ModuleOp mod;

  /// The block the rest of the original block was split off into.
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

mlir::Value AMDGPUPrintfLowering::fitArgInto64Bits(mlir::Value arg) {
  mlir::Type ty = arg.getType();

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    switch (intTy.getWidth()) {
    case 32:
      return mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(),
                                        rewriter.getI64Type(), arg);
    case 64:
      return arg;
    }
  }

  if (ty.isF64())
    return mlir::LLVM::BitcastOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), arg);

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(ty))
    return mlir::LLVM::PtrToIntOp::create(rewriter, op.getLoc(),
                                          rewriter.getI64Type(), arg);

  llvm_unreachable("argument types are checked before lowering");
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
  mlir::Type ptrTy = str.getType();

  // The length is either zero for a null pointer, or the computed value for an
  // actual string. The join block's argument represents the final value.
  //
  // Strictly speaking, the zero does not matter since
  // __ockl_printf_append_string_n ignores the length if the pointer is null.
  mlir::Block *whileBlock =
      rewriter.createBlock(region, cont->getIterator(), {ptrTy}, {op.getLoc()});
  mlir::Block *whileDone = rewriter.createBlock(region, cont->getIterator());
  mlir::Block *join = rewriter.createBlock(
      region, cont->getIterator(), {rewriter.getI64Type()}, {op.getLoc()});

  // Emit an early return for when the pointer is null.
  rewriter.setInsertionPointToEnd(prev);
  mlir::Value zero = getI64(0);
  mlir::Value cmpNull = mlir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, str,
      mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy));
  mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmpNull, join, zero,
                               whileBlock, str);

  // Entry to the while loop.
  rewriter.setInsertionPointToEnd(whileBlock);
  mlir::Value ptrPhi = whileBlock->getArgument(0);
  mlir::Value ptrNext = mlir::LLVM::GEPOp::create(
      rewriter, op.getLoc(), ptrTy, rewriter.getI8Type(), ptrPhi,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{getI64(1)});

  // Condition for the while loop.
  mlir::Value data = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                                rewriter.getI8Type(), ptrPhi);
  mlir::Value cmp = mlir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, data,
      mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                     rewriter.getI8Type(), llvm::APInt(8, 0)));
  mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmp, whileDone,
                               mlir::ValueRange(), whileBlock, ptrNext);

  // Add one to the computed length.
  rewriter.setInsertionPointToEnd(whileDone);
  auto addrTy = mlir::IntegerType::get(rewriter.getContext(),
                                       *dataLayout.getTypeIndexBitwidth(ptrTy));
  mlir::Value endAddr =
      mlir::LLVM::PtrToAddrOp::create(rewriter, op.getLoc(), addrTy, ptrPhi);
  mlir::Value beginAddr =
      mlir::LLVM::PtrToAddrOp::create(rewriter, op.getLoc(), addrTy, str);
  mlir::Value len =
      mlir::LLVM::SubOp::create(rewriter, op.getLoc(), endAddr, beginAddr);
  if (addrTy != rewriter.getI64Type())
    len = mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(),
                                     rewriter.getI64Type(), len);
  len = mlir::LLVM::AddOp::create(rewriter, op.getLoc(), len, getI64(1));

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
  auto name = mlir::StringAttr::get(rewriter.getContext(),
                                    "__ockl_printf_append_string_n");
  // As in OGCG, the declaration takes the pointer type of the first string it
  // is called with. Unlike an LLVM IR call, llvm.call must match its callee, so
  // cast strings from another address space (CIR string literals on SPIR-V are
  // not in the generic address space).
  if (auto fn =
          symbolTables.lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(mod, name)) {
    mlir::Type strTy = fn.getFunctionType().getParamType(1);
    if (str.getType() != strTy)
      str = mlir::LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), strTy,
                                                str);
  }
  return createCall(name, rewriter.getI64Type(),
                    {desc, str, length, isLastInt32});
}

mlir::Value AMDGPUPrintfLowering::appendString(mlir::Value desc,
                                               mlir::Value arg, bool isLast) {
  mlir::Value length = getStrlenWithNull(arg);
  return callAppendStringN(desc, arg, length, isLast);
}

mlir::Value AMDGPUPrintfLowering::processArg(mlir::Value desc, mlir::Value arg,
                                             bool specIsCString, bool isLast) {
  if (specIsCString && mlir::isa<mlir::LLVM::LLVMPointerType>(arg.getType()))
    return appendString(desc, arg, isLast);
  // If the format specifies a string but the argument is not, the frontend will
  // have printed a warning. We just rely on undefined behaviour and send the
  // argument anyway.
  return appendArg(desc, arg, isLast);
}

mlir::Value AMDGPUPrintfLowering::emitHostcall(mlir::ValueRange args) {
  size_t numOps = args.size();
  mlir::Value desc =
      createCall("__ockl_printf_begin", rewriter.getI64Type(), getI64(0));
  desc = appendString(desc, args[0], numOps == 1);

  // FIXME: This invokes hostcall once for each argument. We can pack up to
  // seven scalar printf arguments in a single hostcall. See the signature of
  // callAppendArgs().
  for (size_t i = 1; i != numOps; ++i) {
    bool isLast = i == numOps - 1;
    bool isCString = specIsCString.test(i);
    desc = processArg(desc, args[i], isCString, isLast);
  }

  return mlir::LLVM::TruncOp::create(rewriter, op.getLoc(),
                                     rewriter.getI32Type(), desc);
}

// Align the computed length to next 8 byte boundary.
mlir::Value AMDGPUPrintfLowering::alignTo8(mlir::Value len) {
  mlir::Value tempAdd =
      mlir::LLVM::AddOp::create(rewriter, op.getLoc(), len, getI64(7));
  // OGCG masks with the 32-bit ~7U zero-extended to i64; match it exactly.
  return mlir::LLVM::AndOp::create(rewriter, op.getLoc(), tempAdd, getI64(~7U));
}

// Calculates frame size required for current printf expansion and allocates
// space on printf buffer. Printf frame includes following contents
// [ ControlDWord , format string/Hash , Arguments (each aligned to 8 byte) ]
mlir::Value AMDGPUPrintfLowering::callBufferedPrintfStart(
    mlir::ValueRange args, bool isConstFmtStr,
    llvm::SmallVectorImpl<StringData> &stringContents, mlir::Value &argSize) {
  mlir::Value nonConstStrLen;

  // First 4 bytes to be reserved for control dword
  uint64_t bufSize = 4;
  if (isConstFmtStr) {
    // First 8 bytes of MD5 hash
    bufSize += 8;
  } else {
    mlir::Value lenWithNull = getStrlenWithNull(args[0]);
    nonConstStrLen = alignTo8(lenWithNull);
    stringContents.push_back({"", lenWithNull, nonConstStrLen, false});
  }

  mlir::OperandRange cirArgs = op->getOperands();
  for (size_t i = 1; i < args.size(); i++) {
    if (isCStringArg(args, i)) {
      std::string argStr;
      if (getConstantStringInfo(cirArgs[i], mod, symbolTables, argStr)) {
        bufSize += llvm::alignTo(argStr.size() + 1, 8);
        stringContents.push_back({argStr, {}, {}, true});
      } else {
        mlir::Value lenWithNull = getStrlenWithNull(args[i]);
        mlir::Value lenWithNullAligned = alignTo8(lenWithNull);

        if (nonConstStrLen)
          nonConstStrLen = mlir::LLVM::AddOp::create(
              rewriter, op.getLoc(), lenWithNullAligned, nonConstStrLen);
        else
          nonConstStrLen = lenWithNullAligned;

        stringContents.push_back({"", lenWithNull, lenWithNullAligned, false});
      }
    } else {
      uint64_t allocSize =
          dataLayout.getTypeSize(args[i].getType()).getFixedValue();
      // We end up expanding non string arguments to 8 bytes (args smaller than
      // 8 bytes)
      bufSize += std::max<uint64_t>(allocSize, 8);
    }
  }

  // calculate final size value to be passed to printf_alloc
  if (nonConstStrLen)
    argSize = mlir::LLVM::TruncOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(),
        mlir::LLVM::AddOp::create(rewriter, op.getLoc(), nonConstStrLen,
                                  getI64(bufSize)));
  else
    argSize = getI32(bufSize);

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
  whatToStore.reserve(whatToStore.size() + words.size());
  for (uint32_t word : words)
    whatToStore.push_back(getI32(word));
}

mlir::Value AMDGPUPrintfLowering::processNonStringArg(mlir::Value arg) {
  mlir::Type ty = arg.getType();

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty))
    if (intTy.getWidth() < 64)
      return mlir::LLVM::ZExtOp::create(rewriter, op.getLoc(),
                                        rewriter.getI64Type(), arg);

  if (mlir::isa<mlir::FloatType>(ty))
    if (dataLayout.getTypeSize(ty).getFixedValue() < 8)
      return mlir::LLVM::FPExtOp::create(rewriter, op.getLoc(),
                                         rewriter.getF64Type(), arg);

  return arg;
}

void AMDGPUPrintfLowering::callBufferedPrintfArgPush(
    mlir::ValueRange args, mlir::Value ptrToStore,
    llvm::ArrayRef<StringData> stringContents, bool isConstFmtStr) {
  mlir::Type ptrTy = ptrToStore.getType();
  auto gepInBounds = [&](mlir::Value ptr, mlir::LLVM::GEPArg offset) {
    return mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy,
                                     rewriter.getI8Type(), ptr,
                                     llvm::ArrayRef<mlir::LLVM::GEPArg>{offset},
                                     mlir::LLVM::GEPNoWrapFlags::inbounds);
  };

  const StringData *strIt = stringContents.begin();
  size_t i = isConstFmtStr ? 1 : 0;
  for (; i < args.size(); i++) {
    llvm::SmallVector<mlir::Value, 32> whatToStore;
    if ((i == 0) || isCStringArg(args, i)) {
      if (strIt->isConst) {
        processConstantStringArg(*strIt, whatToStore);
        strIt++;
      } else {
        // This copies the contents of the string, however the next offset
        // is at aligned length, the extra space that might be created due
        // to alignment padding is not populated with any specific value
        // here. This would be safe as long as runtime is sync with
        // the offsets.
        mlir::LLVM::MemcpyOp::create(rewriter, op.getLoc(), ptrToStore, args[i],
                                     strIt->realSize, /*isVolatile=*/false);
        ptrToStore = gepInBounds(ptrToStore, strIt->alignedSize);

        // done with current argument, move to next
        strIt++;
        continue;
      }
    } else {
      whatToStore.push_back(processNonStringArg(args[i]));
    }

    for (mlir::Value toStore : whatToStore) {
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), toStore, ptrToStore);
      ptrToStore = gepInBounds(
          ptrToStore,
          dataLayout.getTypeSize(toStore.getType()).getFixedValue());
    }
  }
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

mlir::Value AMDGPUPrintfLowering::emitBuffered(mlir::ValueRange args) {
  llvm::SmallVector<StringData, 8> stringContents;
  bool isConstFmtStr = !fmtStr.empty();

  mlir::Value argSize;
  mlir::Value ptr =
      callBufferedPrintfStart(args, isConstFmtStr, stringContents, argSize);

  // The buffered version still follows OpenCL printf standards for
  // printf return value, i.e 0 on success, -1 on failure.
  mlir::Value cmp = mlir::LLVM::ICmpOp::create(
      rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::ne, ptr,
      mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptr.getType()));

  // The continuation block doubles as the end block.
  mlir::Block *prev = rewriter.getInsertionBlock();
  mlir::Block *argPush =
      rewriter.createBlock(cont->getParent(), cont->getIterator());
  rewriter.setInsertionPointToEnd(prev);
  mlir::LLVM::CondBrOp::create(rewriter, op.getLoc(), cmp, argPush, cont);
  rewriter.setInsertionPointToEnd(argPush);

  // Create controlDWord and store as the first entry, format as follows
  // Bit 0 (LSB) -> stream (1 if stderr, 0 if stdout, printf always outputs to
  // stdout) Bit 1 -> constant format string (1 if constant) Bits 2-31 -> size
  // of printf data frame
  mlir::Value controlDWord =
      mlir::LLVM::ShlOp::create(rewriter, op.getLoc(), argSize, getI32(2));
  if (isConstFmtStr)
    controlDWord = mlir::LLVM::OrOp::create(rewriter, op.getLoc(), controlDWord,
                                            getI32(2));

  mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), controlDWord, ptr);

  ptr = mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptr.getType(),
                                  rewriter.getI8Type(), ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{4},
                                  mlir::LLVM::GEPNoWrapFlags::inbounds);

  // Create MD5 hash for constant format string, push low 64 bits of the
  // same onto buffer and metadata.
  if (isConstFmtStr) {
    addPrintfFormatMetadata(llvm::getAMDGPUPrintfFormatMetadata(fmtStr),
                            /*onlyIfEmpty=*/false);

    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(),
                                getI64(llvm::getAMDGPUPrintfFormatHash(fmtStr)),
                                ptr);
    ptr = mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptr.getType(),
                                    rewriter.getI8Type(), ptr,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{8},
                                    mlir::LLVM::GEPNoWrapFlags::inbounds);
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
  mlir::LLVM::BrOp::create(rewriter, op.getLoc(), cont);
  rewriter.setInsertionPointToStart(cont);
  mlir::Value notCmp = mlir::LLVM::XOrOp::create(
      rewriter, op.getLoc(), cmp,
      mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                     rewriter.getI1Type(), llvm::APInt(1, 1)));
  return mlir::LLVM::SExtOp::create(rewriter, op.getLoc(),
                                    rewriter.getI32Type(), notCmp);
}

mlir::LogicalResult AMDGPUPrintfLowering::lower(mlir::ValueRange args,
                                                bool isBuffered) {
  // Hostcall passes every argument as a 64-bit word, and only knows how to
  // widen the types the default argument promotions produce.
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

  if (getConstantStringInfo(op.getFormat(), mod, symbolTables, fmtStr))
    llvm::locateAMDGPUPrintfCStrings(specIsCString, fmtStr);
  else
    fmtStr.clear();

  mlir::Block *block = op->getBlock();
  cont = rewriter.splitBlock(block, mlir::Block::iterator(op));
  rewriter.setInsertionPointToEnd(block);

  mlir::Value result;
  if (isBuffered) {
    result = emitBuffered(args);
  } else {
    result = emitHostcall(args);
    mlir::LLVM::BrOp::create(rewriter, op.getLoc(), cont);
  }

  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMOffloadPrintfOpLowering::matchAndRewrite(
    cir::OffloadPrintfOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  llvm::StringRef tripleStr;
  if (auto tripleAttr = mod->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getTripleAttrName()))
    tripleStr = tripleAttr.getValue();
  llvm::Triple triple(tripleStr);

  if (triple.isAMDGCN() ||
      (triple.isSPIRV() && triple.getVendor() == llvm::Triple::AMD)) {
    bool isBuffered = false;
    if (auto kind = mod->getAttrOfType<mlir::StringAttr>(
            cir::CIRDialect::getAMDGPUPrintfKindAttrName()))
      isBuffered = kind.getValue() == "buffered";
    return AMDGPUPrintfLowering(op, rewriter, dataLayout, symbolTables)
        .lower(adaptor.getOperands(), isBuffered);
  }

  return op.emitError() << "offload printf lowering is NYI for target '"
                        << tripleStr << "'";
}

} // namespace direct
} // namespace cir
