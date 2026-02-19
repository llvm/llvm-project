//===-- MIFOpConversion.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/MIFOpConversion.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/MIFCommon.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Runtime/coarray.h"
#include "flang/Runtime/stop.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_MIFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;
using namespace Fortran::runtime;

namespace {

// Default prefix for subroutines of PRIF compiled with LLVM
static std::string getPRIFProcName(std::string fmt) {
  std::ostringstream oss;
  oss << "prif_" << fmt;
  return fir::NameUniquer::doProcedure({"prif"}, {}, oss.str());
}

static mlir::Type getPRIFStatType(fir::FirOpBuilder &builder) {
  return builder.getRefType(builder.getI32Type());
}

static mlir::Type getPRIFErrmsgType(fir::FirOpBuilder &builder) {
  return fir::BoxType::get(fir::CharacterType::get(
      builder.getContext(), 1, fir::CharacterType::unknownLen()));
}

static mlir::Type
genBoxedSequenceType(mlir::Type eleTy,
                     std::optional<int64_t> rank = std::nullopt) {
  if (rank.has_value())
    return fir::BoxType::get(fir::SequenceType::get({rank.value()}, eleTy));
  return fir::BoxType::get(
      fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, eleTy));
}

static mlir::Type getCoarrayHandleType(fir::FirOpBuilder &builder,
                                       mlir::Location loc) {
  // Defining the coarray handle type
  std::string handleDTName =
      fir::NameUniquer::doType({"prif"}, {}, 0, "prif_coarray_handle", {});
  fir::RecordType handleTy =
      fir::RecordType::get(builder.getContext(), handleDTName);
  mlir::Type infoTy =
      fir::BoxType::get(fir::PointerType::get(builder.getNoneType()));
  handleTy.finalize({}, {{"info", infoTy}});

  // Checking if the type information was generated
  fir::TypeInfoOp dt;
  fir::RecordType parentType{};
  mlir::OpBuilder::InsertPoint insertPointIfCreated;
  std::tie(dt, insertPointIfCreated) =
      builder.createTypeInfoOp(loc, handleTy, parentType);
  if (insertPointIfCreated.isSet()) {
    // fir.type_info wasn't built in a previous call.
    dt->setAttr(dt.getNoInitAttrName(), builder.getUnitAttr());
    dt->setAttr(dt.getNoDestroyAttrName(), builder.getUnitAttr());
    dt->setAttr(dt.getNoFinalAttrName(), builder.getUnitAttr());
    builder.restoreInsertionPoint(insertPointIfCreated);
    // Create global op
    // FIXME: replace handleTy by the Derived type that describe handleTy
    std::string globalName =
        fir::NameUniquer::getTypeDescriptorName(handleDTName);
    auto linkage = builder.createLinkOnceODRLinkage();
    builder.createGlobal(loc, handleTy, globalName, linkage);
  }
  return handleTy;
}

mlir::Value getCoarrayHandle(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value coarray) {
  mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  std::string uniqName = mif::getFullUniqName(coarray);
  if (!uniqName.empty()) {
    std::string globalName = uniqName + coarrayHandleSuffix.str();
    mlir::SymbolRefAttr symAttr =
        mlir::SymbolRefAttr::get(builder.getContext(), globalName);
    mlir::Value coarrayHandle =
        fir::AddrOfOp::create(builder, loc, builder.getRefType(boxTy), symAttr);
    return fir::LoadOp::create(builder, loc, coarrayHandle);
  }
  mlir::emitError(coarray.getLoc(),
                  "Unable to locate the coarray handle for this argument.");
  return mlir::Value{};
}

// Function to generate the PRIF runtime function call to retrieve
// the number of images in the current team
static mlir::Value getNumImages(fir::FirOpBuilder &builder,
                                mlir::Location loc) {
  mlir::Type i32Ty = builder.getI32Type();
  mlir::Value result = builder.createTemporary(loc, i32Ty);
  mlir::FunctionType ftype = mlir::FunctionType::get(
      builder.getContext(),
      /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
  mlir::func::FuncOp funcOp =
      builder.createFunction(loc, getPRIFProcName("num_images"), ftype);
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, ftype, result);
  fir::CallOp::create(builder, loc, funcOp, args);
  return fir::LoadOp::create(builder, loc, result);
}

static std::pair<mlir::Value, mlir::Value>
genCoBounds(fir::FirOpBuilder &builder, mlir::Location loc,
            mif::AllocCoarrayOp op) {
  mlir::Value ucobounds, lcobounds;
  mlir::DenseI64ArrayAttr lcbsAttr = op.getLcoboundsAttr();
  mlir::DenseI64ArrayAttr ucbsAttr = op.getUcoboundsAttr();

  size_t corank = lcbsAttr.size();
  mlir::Type i64Ty = builder.getI64Type();
  mlir::Type addrType = builder.getRefType(i64Ty);
  mlir::Type arrayType = fir::SequenceType::get(
      {static_cast<fir::SequenceType::Extent>(corank)}, i64Ty);
  lcobounds = builder.createTemporary(loc, arrayType);
  ucobounds = builder.createTemporary(loc, arrayType);

  for (size_t i = 0; i < corank; i++) {
    auto index = builder.createIntegerConstant(loc, builder.getIndexType(), i);
    // Lower cobounds
    auto lcovalue = builder.createIntegerConstant(loc, i64Ty, lcbsAttr[i]);
    auto lcoaddr =
        fir::CoordinateOp::create(builder, loc, addrType, lcobounds, index);
    fir::StoreOp::create(builder, loc, lcovalue, lcoaddr);

    // Upper cobounds
    auto ucovalue = builder.createIntegerConstant(loc, i64Ty, ucbsAttr[i]);
    auto ucoaddr =
        fir::CoordinateOp::create(builder, loc, addrType, ucobounds, index);
    fir::StoreOp::create(builder, loc, ucovalue, ucoaddr);
  }

  lcobounds = builder.createBox(loc, lcobounds);
  ucobounds = builder.createBox(loc, ucobounds);

  // Computing last ucobound
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(ComputeLastUcobound)>(loc, builder);
  mlir::Value numImages = getNumImages(builder, loc);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, func.getFunctionType(), numImages, lcobounds, ucobounds);
  fir::CallOp::create(builder, loc, func, args);

  return {lcobounds, ucobounds};
}

// Storing the coarray descriptor as a global variable
void storeCoarrayHandle(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Value coarrayHandle, std::string uniqName) {
  std::string globalName = uniqName + coarrayHandleSuffix.str();
  fir::GlobalOp global = builder.getNamedGlobal(globalName);
  if (!global) {
    global = builder.createGlobal(loc, coarrayHandle.getType(), globalName,
                                  builder.createLinkOnceLinkage());
    mlir::Region &region = global.getRegion();
    region.push_back(new mlir::Block);
    mlir::Block &block = region.back();
    auto insertPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&block);
    auto box = fir::factory::createUnallocatedBox(builder, loc,
                                                  coarrayHandle.getType(), {});
    fir::HasValueOp::create(builder, loc, box);
    builder.restoreInsertionPoint(insertPt);
  }

  mlir::SymbolRefAttr symAttr =
      mlir::SymbolRefAttr::get(builder.getContext(), globalName);
  auto addrOf = fir::AddrOfOp::create(
      builder, loc, builder.getRefType(coarrayHandle.getType()), symAttr);
  fir::StoreOp::create(builder, loc, coarrayHandle, addrOf);
}

static int computeElementByteSize(mlir::Location loc, mlir::Type type,
                                  fir::KindMapping &kindMap,
                                  bool emitErrorOnFailure = true) {
  auto eleTy = fir::unwrapSequenceType(type);
  if (auto t{mlir::dyn_cast<mlir::IntegerType>(eleTy)})
    return t.getWidth() / 8;
  if (auto t{mlir::dyn_cast<mlir::FloatType>(eleTy)})
    return t.getWidth() / 8;
  if (auto t{mlir::dyn_cast<fir::LogicalType>(eleTy)})
    return kindMap.getLogicalBitsize(t.getFKind()) / 8;
  if (auto t{mlir::dyn_cast<mlir::ComplexType>(eleTy)}) {
    int elemSize =
        mlir::cast<mlir::FloatType>(t.getElementType()).getWidth() / 8;
    return 2 * elemSize;
  }
  if (auto t{mlir::dyn_cast<fir::CharacterType>(eleTy)})
    return kindMap.getCharacterBitsize(t.getFKind()) / 8;
  if (emitErrorOnFailure)
    mlir::emitError(loc, "unsupported type");
  return 0;
}

// Function used to compute the size in bytes of an entity. This function
// is used during an allocation of a coarray (or a component of a coarray),
// as it's a required argument in some PRIF procedures.
static mlir::Value getSizeInBytes(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::ModuleOp module,
                                  mlir::DataLayout *dl,
                                  const fir::LLVMTypeConverter *typeConverter,
                                  mlir::Value box) {
  fir::KindMapping kindMap{fir::getKindMapping(module)};
  mlir::Type baseTy = fir::unwrapPassByRefType(box.getType());

  mlir::Value sizeInBytes = builder.createTemporary(loc, builder.getI64Type());
  mlir::Value bytes;
  if (!mlir::dyn_cast_or_null<fir::BaseBoxType>(baseTy)) {
    if (fir::isa_trivial(baseTy)) {
      int width = computeElementByteSize(loc, baseTy, kindMap);
      bytes = builder.createIntegerConstant(loc, builder.getI64Type(), width);
    } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(baseTy)) {
      std::size_t size = 0;
      if (fir::isa_derived(seqTy.getEleTy())) {
        mlir::Type structTy = typeConverter->convertType(seqTy.getEleTy());
        size = dl->getTypeSizeInBits(structTy) / 8;
      } else {
        size = computeElementByteSize(loc, seqTy.getEleTy(), kindMap);
      }
      mlir::Value width =
          builder.createIntegerConstant(loc, builder.getI64Type(), size);
      mlir::Value nbElem;
      if (fir::sequenceWithNonConstantShape(seqTy)) {
        // TODO: Not handle for now, but will be do it later.
        mlir::emitError(loc,
                        "unsupported sequence type with non constant shape");
      } else {
        nbElem = builder.createIntegerConstant(loc, builder.getI64Type(),
                                               seqTy.getConstantArraySize());
      }
      bytes = mlir::arith::MulIOp::create(builder, loc, nbElem, width);
    } else if (fir::isa_derived(baseTy)) {
      mlir::Type structTy = typeConverter->convertType(baseTy);
      std::size_t structSize = dl->getTypeSizeInBits(structTy) / 8;
      bytes =
          builder.createIntegerConstant(loc, builder.getI64Type(), structSize);
    } else if (fir::isa_char(baseTy)) {
      mlir::Type charTy = typeConverter->convertType(baseTy);
      std::size_t charSize = dl->getTypeSizeInBits(charTy) / 8;
      bytes =
          builder.createIntegerConstant(loc, builder.getI64Type(), charSize);
    } else {
      mlir::emitError(loc, "unsupported type in mif allocation\n");
    }
  } else {
    if (fir::isa_ref_type(box.getType()))
      box = fir::LoadOp::create(builder, loc, box);
    bytes = fir::BoxEleSizeOp::create(builder, loc, builder.getI64Type(), box);
    auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(baseTy);
    if (fir::extractSequenceType(boxTy)) {
      mlir::Value extent = builder.createConvert(
          loc, builder.getI64Type(), fir::runtime::genSize(builder, loc, box));
      bytes = mlir::arith::MulIOp::create(builder, loc, bytes, extent);
    }
  }
  fir::StoreOp::create(builder, loc, bytes, sizeInBytes);
  return sizeInBytes;
}

// Most PRIF functions take `errmsg` and `errmsg_alloc` as two optional
// arguments of intent (out). One is allocatable, the other is not.
// It is the responsibility of the compiler to ensure that the appropriate
// optional argument is passed, and at most one must be provided in a given
// call.
// Depending on the type of `errmsg`, this function will return the pair
// corresponding to (`errmsg`, `errmsg_alloc`).
static std::pair<mlir::Value, mlir::Value>
genErrmsgPRIF(fir::FirOpBuilder &builder, mlir::Location loc,
              mlir::Value errmsg) {
  mlir::Value absent =
      fir::AbsentOp::create(builder, loc, getPRIFErrmsgType(builder));
  if (!errmsg)
    return {absent, absent};

  bool isAllocatableErrmsg = fir::isAllocatableType(errmsg.getType());
  mlir::Value errMsg = isAllocatableErrmsg ? absent : errmsg;
  mlir::Value errMsgAlloc = isAllocatableErrmsg ? errmsg : absent;
  return {errMsg, errMsgAlloc};
}

static mlir::Value genStatPRIF(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value stat) {
  if (!stat)
    return fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
  return stat;
}

static fir::CallOp genPRIFStopErrorStop(fir::FirOpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value stopCode,
                                        bool isError = false) {
  mlir::Type stopCharTy = fir::BoxCharType::get(builder.getContext(), 1);
  mlir::Type i1Ty = builder.getI1Type();
  mlir::Type i32Ty = builder.getI32Type();

  mlir::FunctionType ftype = mlir::FunctionType::get(
      builder.getContext(),
      /*inputs*/
      {builder.getRefType(i1Ty), builder.getRefType(i32Ty), stopCharTy},
      /*results*/ {});
  mlir::func::FuncOp funcOp =
      isError
          ? builder.createFunction(loc, getPRIFProcName("error_stop"), ftype)
          : builder.createFunction(loc, getPRIFProcName("stop"), ftype);

  // QUIET is managed in flang-rt, so its value is set to TRUE here.
  mlir::Value q = builder.createBool(loc, true);
  mlir::Value quiet = builder.createTemporary(loc, i1Ty);
  fir::StoreOp::create(builder, loc, q, quiet);

  mlir::Value stopCodeInt, stopCodeChar;
  if (!stopCode) {
    stopCodeChar = fir::AbsentOp::create(builder, loc, stopCharTy);
    stopCodeInt =
        fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  } else if (fir::isa_integer(stopCode.getType())) {
    stopCodeChar = fir::AbsentOp::create(builder, loc, stopCharTy);
    stopCodeInt = builder.createTemporary(loc, i32Ty);
    if (stopCode.getType() != i32Ty)
      stopCode = fir::ConvertOp::create(builder, loc, i32Ty, stopCode);
    fir::StoreOp::create(builder, loc, stopCode, stopCodeInt);
  } else {
    stopCodeChar = stopCode;
    if (!mlir::isa<fir::BoxCharType>(stopCodeChar.getType())) {
      auto len =
          fir::UndefOp::create(builder, loc, builder.getCharacterLengthType());
      stopCodeChar =
          fir::EmboxCharOp::create(builder, loc, stopCharTy, stopCodeChar, len);
    }
    stopCodeInt =
        fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  }

  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, quiet, stopCodeInt, stopCodeChar);
  return fir::CallOp::create(builder, loc, funcOp, args);
}

enum class TerminationKind { Normal = 0, Error = 1, FailImage = 2 };
// Generates a wrapper function for the different kind of termination in PRIF.
// This function will be used to register wrappers on PRIF runtime termination
// functions into the Fortran runtime.
mlir::Value genTerminationOperationWrapper(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::ModuleOp module,
                                           TerminationKind termKind) {
  std::string funcName;
  mlir::FunctionType funcType =
      mlir::FunctionType::get(builder.getContext(), {}, {});
  mlir::Type i32Ty = builder.getI32Type();
  if (termKind == TerminationKind::Normal) {
    funcName = getPRIFProcName("stop");
    funcType = mlir::FunctionType::get(builder.getContext(), {i32Ty}, {});
  } else if (termKind == TerminationKind::Error) {
    funcName = getPRIFProcName("error_stop");
    funcType = mlir::FunctionType::get(builder.getContext(), {i32Ty}, {});
  } else {
    funcName = getPRIFProcName("fail_image");
  }
  funcName += "_termination_wrapper";
  mlir::func::FuncOp funcWrapperOp =
      module.lookupSymbol<mlir::func::FuncOp>(funcName);

  if (!funcWrapperOp) {
    funcWrapperOp = builder.createFunction(loc, funcName, funcType);

    // generating the body of the function.
    mlir::OpBuilder::InsertPoint saveInsertPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(funcWrapperOp.addEntryBlock());

    if (termKind == TerminationKind::Normal) {
      genPRIFStopErrorStop(builder, loc, funcWrapperOp.getArgument(0),
                           /*isError*/ false);
    } else if (termKind == TerminationKind::Error) {
      genPRIFStopErrorStop(builder, loc, funcWrapperOp.getArgument(0),
                           /*isError*/ true);
    } else {
      mlir::func::FuncOp fOp = builder.createFunction(
          loc, getPRIFProcName("fail_image"),
          mlir::FunctionType::get(builder.getContext(), {}, {}));
      fir::CallOp::create(builder, loc, fOp);
    }

    mlir::func::ReturnOp::create(builder, loc);
    builder.restoreInsertionPoint(saveInsertPoint);
  }

  mlir::SymbolRefAttr symbolRef = mlir::SymbolRefAttr::get(
      builder.getContext(), funcWrapperOp.getSymNameAttr());
  return fir::AddrOfOp::create(builder, loc, funcType, symbolRef);
}

/// Convert mif.init operation to runtime call of 'prif_init'
struct MIFInitOpConversion : public mlir::OpRewritePattern<mif::InitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::InitOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value result = builder.createTemporary(loc, i32Ty);

    // Registering PRIF runtime termination to the Fortran runtime
    // STOP
    mlir::Value funcStopOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::Normal);
    mlir::func::FuncOp normalEndFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterImagesNormalEndCallback)>(
            loc, builder);
    llvm::SmallVector<mlir::Value> args1 = fir::runtime::createArguments(
        builder, loc, normalEndFunc.getFunctionType(), funcStopOp);
    fir::CallOp::create(builder, loc, normalEndFunc, args1);

    // ERROR STOP
    mlir::Value funcErrorStopOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::Error);
    mlir::func::FuncOp errorFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterImagesErrorCallback)>(
            loc, builder);
    llvm::SmallVector<mlir::Value> args2 = fir::runtime::createArguments(
        builder, loc, errorFunc.getFunctionType(), funcErrorStopOp);
    fir::CallOp::create(builder, loc, errorFunc, args2);

    // FAIL IMAGE
    mlir::Value failImageOp = genTerminationOperationWrapper(
        builder, loc, mod, TerminationKind::FailImage);
    mlir::func::FuncOp failImageFunc =
        fir::runtime::getRuntimeFunc<mkRTKey(RegisterFailImageCallback)>(
            loc, builder);
    llvm::SmallVector<mlir::Value> args3 = fir::runtime::createArguments(
        builder, loc, errorFunc.getFunctionType(), failImageOp);
    fir::CallOp::create(builder, loc, failImageFunc, args3);

    // Intialize the multi-image parallel environment
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("init"), ftype);
    llvm::SmallVector<mlir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, result);
    fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
    return mlir::success();
  }
};

/// Convert mif.this_image operation to PRIF runtime call
struct MIFThisImageOpConversion
    : public mlir::OpRewritePattern<mif::ThisImageOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::ThisImageOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (op.getCoarray())
      TODO(loc, "mif.this_image op with coarray argument.");
    else {
      mlir::Type i32Ty = builder.getI32Type();
      mlir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
      mlir::Value result = builder.createTemporary(loc, i32Ty);
      mlir::FunctionType ftype = mlir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {boxTy, builder.getRefType(i32Ty)}, /*results*/ {});
      mlir::Value teamArg = op.getTeam();
      if (!op.getTeam())
        teamArg = fir::AbsentOp::create(builder, loc, boxTy);

      mlir::func::FuncOp funcOp = builder.createFunction(
          loc, getPRIFProcName("this_image_no_coarray"), ftype);
      llvm::SmallVector<mlir::Value> args =
          fir::runtime::createArguments(builder, loc, ftype, teamArg, result);
      fir::CallOp::create(builder, loc, funcOp, args);
      rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
      return mlir::success();
    }
  }
};

/// Convert mif.num_images operation to runtime call of
/// prif_num_images_with_{team|team_number}
struct MIFNumImagesOpConversion
    : public mlir::OpRewritePattern<mif::NumImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::NumImagesOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type boxTy = fir::BoxType::get(rewriter.getNoneType());
    mlir::Value result = builder.createTemporary(loc, i32Ty);

    mlir::func::FuncOp funcOp;
    llvm::SmallVector<mlir::Value> args;
    if (!op.getTeam() && !op.getTeamNumber()) {
      mlir::FunctionType ftype = mlir::FunctionType::get(
          builder.getContext(),
          /*inputs*/ {builder.getRefType(i32Ty)}, /*results*/ {});
      funcOp =
          builder.createFunction(loc, getPRIFProcName("num_images"), ftype);
      args = fir::runtime::createArguments(builder, loc, ftype, result);
    } else {
      if (op.getTeam()) {
        mlir::FunctionType ftype =
            mlir::FunctionType::get(builder.getContext(),
                                    /*inputs*/
                                    {boxTy, builder.getRefType(i32Ty)},
                                    /*results*/ {});
        funcOp = builder.createFunction(
            loc, getPRIFProcName("num_images_with_team"), ftype);
        args = fir::runtime::createArguments(builder, loc, ftype, op.getTeam(),
                                             result);
      } else {
        mlir::Value teamNumber = builder.createTemporary(loc, i64Ty);
        mlir::Value cst = op.getTeamNumber();
        if (op.getTeamNumber().getType() != i64Ty)
          cst = fir::ConvertOp::create(builder, loc, i64Ty, op.getTeamNumber());
        fir::StoreOp::create(builder, loc, cst, teamNumber);
        mlir::FunctionType ftype = mlir::FunctionType::get(
            builder.getContext(),
            /*inputs*/ {builder.getRefType(i64Ty), builder.getRefType(i32Ty)},
            /*results*/ {});
        funcOp = builder.createFunction(
            loc, getPRIFProcName("num_images_with_team_number"), ftype);
        args = fir::runtime::createArguments(builder, loc, ftype, teamNumber,
                                             result);
      }
    }
    fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOpWithNewOp<fir::LoadOp>(op, result);
    return mlir::success();
  }
};

/// Convert mif.sync_all operation to runtime call of 'prif_sync_all'
struct MIFSyncAllOpConversion : public mlir::OpRewritePattern<mif::SyncAllOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncAllOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_all"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Convert mif.sync_images operation to runtime call of 'prif_sync_images'
struct MIFSyncImagesOpConversion
    : public mlir::OpRewritePattern<mif::SyncImagesOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncImagesOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type imgSetTy = fir::BoxType::get(fir::SequenceType::get(
        {fir::SequenceType::getUnknownExtent()}, builder.getI32Type()));
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {imgSetTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_images"), ftype);

    // If imageSet is scalar, PRIF require to pass an array of size 1.
    mlir::Value imageSet = op.getImageSet();
    if (!imageSet)
      imageSet = fir::AbsentOp::create(builder, loc, imgSetTy);
    else if (auto boxTy = mlir::dyn_cast<fir::BoxType>(imageSet.getType())) {
      if (!mlir::isa<fir::SequenceType>(boxTy.getEleTy())) {
        mlir::Value one =
            builder.createIntegerConstant(loc, builder.getI32Type(), 1);
        mlir::Value shape = fir::ShapeOp::create(builder, loc, one);
        imageSet =
            fir::ReboxOp::create(builder, loc,
                                 fir::BoxType::get(fir::SequenceType::get(
                                     {1}, builder.getI32Type())),
                                 imageSet, shape, mlir::Value{});
      }
    }
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, imageSet, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Convert mif.sync_memory operation to runtime call of 'prif_sync_memory'
struct MIFSyncMemoryOpConversion
    : public mlir::OpRewritePattern<mif::SyncMemoryOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncMemoryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_memory"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Convert mif.sync_team operation to runtime call of 'prif_sync_team'
struct MIFSyncTeamOpConversion
    : public mlir::OpRewritePattern<mif::SyncTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::SyncTeamOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {boxTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("sync_team"), ftype);

    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, op.getTeam(), stat, errmsgArg, errmsgAllocArg);
    rewriter.replaceOpWithNewOp<fir::CallOp>(op, funcOp, args);
    return mlir::success();
  }
};

/// Generate call to collective subroutines except co_reduce
/// A must be lowered as a box
static fir::CallOp genCollectiveSubroutine(fir::FirOpBuilder &builder,
                                           mlir::Location loc, mlir::Value A,
                                           mlir::Value image, mlir::Value stat,
                                           mlir::Value errmsg,
                                           std::string coName) {
  mlir::Value rootImage;
  mlir::Type i32Ty = builder.getI32Type();
  if (!image)
    rootImage = fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));
  else {
    rootImage = builder.createTemporary(loc, i32Ty);
    if (image.getType() != i32Ty)
      image = fir::ConvertOp::create(builder, loc, i32Ty, image);
    fir::StoreOp::create(builder, loc, image, rootImage);
  }

  mlir::Type errmsgTy = getPRIFErrmsgType(builder);
  mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
  mlir::FunctionType ftype =
      mlir::FunctionType::get(builder.getContext(),
                              /*inputs*/
                              {boxTy, builder.getRefType(builder.getI32Type()),
                               getPRIFStatType(builder), errmsgTy, errmsgTy},
                              /*results*/ {});
  mlir::func::FuncOp funcOp = builder.createFunction(loc, coName, ftype);

  auto [errmsgArg, errmsgAllocArg] = genErrmsgPRIF(builder, loc, errmsg);
  if (!stat)
    stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, ftype, A, rootImage, stat, errmsgArg, errmsgAllocArg);
  return fir::CallOp::create(builder, loc, funcOp, args);
}

/// Convert mif.co_broadcast operation to runtime call of 'prif_co_broadcast'
struct MIFCoBroadcastOpConversion
    : public mlir::OpRewritePattern<mif::CoBroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getSourceImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_broadcast"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_max operation to runtime call of 'prif_co_max'
struct MIFCoMaxOpConversion : public mlir::OpRewritePattern<mif::CoMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoMaxOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp;
    mlir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (mlir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_max"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_min operation to runtime call of 'prif_co_min'
struct MIFCoMinOpConversion : public mlir::OpRewritePattern<mif::CoMinOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoMinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp;
    mlir::Type argTy =
        fir::unwrapSequenceType(fir::unwrapPassByRefType(op.getA().getType()));
    if (mlir::isa<fir::CharacterType>(argTy))
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min_character"));
    else
      callOp = genCollectiveSubroutine(
          builder, loc, op.getA(), op.getResultImage(), op.getStat(),
          op.getErrmsg(), getPRIFProcName("co_min"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.co_sum operation to runtime call of 'prif_co_sum'
struct MIFCoSumOpConversion : public mlir::OpRewritePattern<mif::CoSumOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::CoSumOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    fir::CallOp callOp = genCollectiveSubroutine(
        builder, loc, op.getA(), op.getResultImage(), op.getStat(),
        op.getErrmsg(), getPRIFProcName("co_sum"));
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.form_team operation to runtime call of 'prif_form_team'
struct MIFFormTeamOpConversion
    : public mlir::OpRewritePattern<mif::FormTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::FormTeamOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {builder.getRefType(builder.getI64Type()), boxTy,
         builder.getRefType(builder.getI32Type()), getPRIFStatType(builder),
         errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("form_team"), ftype);

    mlir::Type i64Ty = builder.getI64Type();
    mlir::Value teamNumber = builder.createTemporary(loc, i64Ty);
    mlir::Value t =
        (op.getTeamNumber().getType() == i64Ty)
            ? op.getTeamNumber()
            : fir::ConvertOp::create(builder, loc, i64Ty, op.getTeamNumber());
    fir::StoreOp::create(builder, loc, t, teamNumber);

    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value newIndex;
    if (op.getNewIndex()) {
      newIndex = builder.createTemporary(loc, i32Ty);
      mlir::Value ni =
          (op.getNewIndex().getType() == i32Ty)
              ? op.getNewIndex()
              : fir::ConvertOp::create(builder, loc, i32Ty, op.getNewIndex());
      fir::StoreOp::create(builder, loc, ni, newIndex);
    } else
      newIndex = fir::AbsentOp::create(builder, loc, builder.getRefType(i32Ty));

    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, teamNumber, op.getTeamVar(), newIndex, stat,
        errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.change_team operation to runtime call of 'prif_change_team'
struct MIFChangeTeamOpConversion
    : public mlir::OpRewritePattern<mif::ChangeTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::ChangeTeamOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    builder.setInsertionPoint(op);

    mlir::Location loc = op.getLoc();
    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {boxTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("change_team"), ftype);

    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, op.getTeam(), stat, errmsgArg, errmsgAllocArg);
    fir::CallOp::create(builder, loc, funcOp, args);

    mlir::Operation *changeOp = op.getOperation();
    auto &bodyRegion = op.getRegion();
    mlir::Block &bodyBlock = bodyRegion.front();

    rewriter.inlineBlockBefore(&bodyBlock, changeOp);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Convert mif.end_team operation to runtime call of 'prif_end_team'
struct MIFEndTeamOpConversion : public mlir::OpRewritePattern<mif::EndTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::EndTeamOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/ {getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("end_team"), ftype);

    mlir::Value stat = genStatPRIF(builder, loc, op.getStat());
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, stat, errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

/// Convert mif.get_team operation to runtime call of 'prif_get_team'
struct MIFGetTeamOpConversion : public mlir::OpRewritePattern<mif::GetTeamOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::GetTeamOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::Type lvlTy = builder.getRefType(builder.getI32Type());
    mlir::FunctionType ftype =
        mlir::FunctionType::get(builder.getContext(),
                                /*inputs*/ {lvlTy, boxTy},
                                /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("get_team"), ftype);

    mlir::Value level = op.getLevel();
    if (!level)
      level = fir::AbsentOp::create(builder, loc, lvlTy);
    else {
      mlir::Value cst = op.getLevel();
      mlir::Type i32Ty = builder.getI32Type();
      level = builder.createTemporary(loc, i32Ty);
      if (cst.getType() != i32Ty)
        cst = builder.createConvert(loc, i32Ty, cst);
      fir::StoreOp::create(builder, loc, cst, level);
    }
    mlir::Type resultType = op.getResult().getType();
    mlir::Type baseTy = fir::unwrapRefType(resultType);
    mlir::Value team = builder.createTemporary(loc, baseTy);
    fir::EmboxOp box = fir::EmboxOp::create(builder, loc, resultType, team);

    llvm::SmallVector<mlir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, level, box);
    fir::CallOp::create(builder, loc, funcOp, args);

    rewriter.replaceOp(op, box);
    return mlir::success();
  }
};

/// Convert mif.team_number operation to runtime call of 'prif_team_number'
struct MIFTeamNumberOpConversion
    : public mlir::OpRewritePattern<mif::TeamNumberOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::TeamNumberOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::FunctionType ftype =
        mlir::FunctionType::get(builder.getContext(),
                                /*inputs*/ {boxTy, builder.getRefType(i64Ty)},
                                /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("team_number"), ftype);

    mlir::Value team = op.getTeam();
    if (!team)
      team = fir::AbsentOp::create(builder, loc, boxTy);

    mlir::Value result = builder.createTemporary(loc, i64Ty);
    llvm::SmallVector<mlir::Value> args =
        fir::runtime::createArguments(builder, loc, ftype, team, result);
    fir::CallOp::create(builder, loc, funcOp, args);
    fir::LoadOp load = fir::LoadOp::create(builder, loc, result);
    rewriter.replaceOp(op, load);
    return mlir::success();
  }
};

/// Convert mif.alloca_coarray operation to runtime call of
/// 'prif_allocate_coarray'
struct MIFAllocCoarrayOpConversion
    : public mlir::OpRewritePattern<mif::AllocCoarrayOp> {
  using OpRewritePattern::OpRewritePattern;

  MIFAllocCoarrayOpConversion(mlir::MLIRContext *context, mlir::DataLayout *dl,
                              const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(mif::AllocCoarrayOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type i64Ty = builder.getI64Type();
    mlir::Type ptrTy = fir::PointerType::get(builder.getNoneType());
    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type coboundsTy = genBoxedSequenceType(i64Ty);
    // Type of the procedure pointed by final_func will be the following :
    mlir::Type procTypePtr = fir::BoxProcType::get(
        builder.getContext(),
        mlir::FunctionType::get(builder.getContext(),
                                {boxTy, getPRIFStatType(builder), errmsgTy},
                                {}));
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {coboundsTy, coboundsTy, builder.getRefType(i64Ty),
         builder.getRefType(builder.getNoneType()), boxTy, ptrTy,
         getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp =
        builder.createFunction(loc, getPRIFProcName("allocate_coarray"), ftype);

    // TODO: Handle final_func if needed
    mlir::Value finalFunc = builder.createTemporary(loc, procTypePtr);
    mlir::Value nullBoxProc =
        fir::factory::createNullBoxProc(builder, loc, procTypePtr);
    fir::StoreOp::create(builder, loc, nullBoxProc, finalFunc);
    // Allocate instance of prif_coarray_handle type based on the PRIF
    // specification.
    mlir::Type handleTy = getCoarrayHandleType(builder, loc);
    mlir::Value coarrayHandle =
        builder.createBox(loc, builder.createTemporary(loc, handleTy));

    mlir::Value allocMem = builder.createTemporary(loc, ptrTy);
    mlir::Value addrCvt =
        fir::ConvertOp::create(builder, loc, ptrTy, op.getBox());
    fir::StoreOp::create(builder, loc, addrCvt, allocMem);

    mlir::Value sizeInBytes =
        getSizeInBytes(builder, loc, mod, dl, typeConverter, op.getBox());
    auto [lcobounds, ucobounds] = genCoBounds(builder, loc, op);
    mlir::Value stat = op.getStat();
    if (!stat)
      stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());

    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, lcobounds, ucobounds, sizeInBytes, finalFunc,
        coarrayHandle, allocMem, stat, errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);

    storeCoarrayHandle(builder, loc, coarrayHandle, op.getUniqName().str());

    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }

private:
  mlir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

/// Convert mif.dealloca_coarray operation to runtime call of
/// 'prif_deallocate_coarray'
struct MIFDeallocCoarrayOpConversion
    : public mlir::OpRewritePattern<mif::DeallocCoarrayOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mif::DeallocCoarrayOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->template getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    mlir::Type errmsgTy = getPRIFErrmsgType(builder);
    mlir::Type boxTy = fir::BoxType::get(builder.getNoneType());
    mlir::FunctionType ftype = mlir::FunctionType::get(
        builder.getContext(),
        /*inputs*/
        {boxTy, getPRIFStatType(builder), errmsgTy, errmsgTy},
        /*results*/ {});
    mlir::func::FuncOp funcOp = builder.createFunction(
        loc, getPRIFProcName("deallocate_coarray"), ftype);

    mlir::Value coarrayHandle = getCoarrayHandle(builder, loc, op.getCoarray());
    mlir::Value stat = op.getStat();
    if (!stat)
      stat = fir::AbsentOp::create(builder, loc, getPRIFStatType(builder));
    auto [errmsgArg, errmsgAllocArg] =
        genErrmsgPRIF(builder, loc, op.getErrmsg());
    llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
        builder, loc, ftype, coarrayHandle, stat, errmsgArg, errmsgAllocArg);
    fir::CallOp callOp = fir::CallOp::create(builder, loc, funcOp, args);
    rewriter.replaceOp(op, callOp);
    return mlir::success();
  }
};

class MIFOpConversion : public fir::impl::MIFOpConversionBase<MIFOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    mlir::SymbolTable symtab(module);

    std::optional<mlir::DataLayout> dl = fir::support::getOrSetMLIRDataLayout(
        module, /*allowDefaultLayout=*/false);
    if (!dl.has_value()) {
      mlir::emitError(
          module.getLoc(),
          "data layout attribute is required to perform MIFOpConversion pass");
      return signalPassFailure();
    }

    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    mif::populateMIFOpConversionPatterns(typeConverter, *dl, patterns);

    target.addLegalDialect<fir::FIROpsDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in MIF op conversion\n");
      return signalPassFailure();
    }
  }
};
} // namespace

void mif::populateMIFOpConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::DataLayout &dl,
    mlir::RewritePatternSet &patterns) {
  patterns.insert<MIFAllocCoarrayOpConversion>(patterns.getContext(), &dl,
                                               &converter);
  patterns.insert<
      MIFInitOpConversion, MIFThisImageOpConversion, MIFNumImagesOpConversion,
      MIFSyncAllOpConversion, MIFSyncImagesOpConversion,
      MIFSyncMemoryOpConversion, MIFSyncTeamOpConversion,
      MIFCoBroadcastOpConversion, MIFCoMaxOpConversion, MIFCoMinOpConversion,
      MIFCoSumOpConversion, MIFFormTeamOpConversion, MIFChangeTeamOpConversion,
      MIFEndTeamOpConversion, MIFGetTeamOpConversion, MIFTeamNumberOpConversion,
      MIFDeallocCoarrayOpConversion>(patterns.getContext());
}
