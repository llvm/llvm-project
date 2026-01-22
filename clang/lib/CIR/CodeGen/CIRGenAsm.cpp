//===--- CIRGenAsm.cpp - Inline Assembly Support for CIR CodeGen ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to emit inline assembly.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticSema.h"

#include "CIRGenFunction.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static AsmFlavor inferFlavor(const CIRGenModule &cgm, const AsmStmt &s) {
  AsmFlavor gnuAsmFlavor =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmFlavor::x86_att
          : AsmFlavor::x86_intel;

  return isa<MSAsmStmt>(&s) ? AsmFlavor::x86_intel : gnuAsmFlavor;
}

static void collectClobbers(const CIRGenFunction &cgf, const AsmStmt &s,
                            std::string &constraints, bool &hasUnwindClobber,
                            bool &readOnly, bool readNone) {

  hasUnwindClobber = false;
  const CIRGenModule &cgm = cgf.getCIRGenModule();

  // Clobbers
  for (unsigned i = 0, e = s.getNumClobbers(); i != e; i++) {
    std::string clobber = s.getClobber(i);
    if (clobber == "memory") {
      readOnly = readNone = false;
    } else if (clobber == "unwind") {
      hasUnwindClobber = true;
      continue;
    } else if (clobber != "cc") {
      clobber = cgf.getTarget().getNormalizedGCCRegisterName(clobber);
      if (cgm.getCodeGenOpts().StackClashProtector &&
          cgf.getTarget().isSPRegName(clobber))
        cgm.getDiags().Report(s.getAsmLoc(),
                              diag::warn_stack_clash_protection_inline_asm);
    }

    if (isa<MSAsmStmt>(&s)) {
      if (clobber == "eax" || clobber == "edx") {
        if (constraints.find("=&A") != std::string::npos)
          continue;
        std::string::size_type position1 =
            constraints.find("={" + clobber + "}");
        if (position1 != std::string::npos) {
          constraints.insert(position1 + 1, "&");
          continue;
        }
        std::string::size_type position2 = constraints.find("=A");
        if (position2 != std::string::npos) {
          constraints.insert(position2 + 1, "&");
          continue;
        }
      }
    }
    if (!constraints.empty())
      constraints += ',';

    constraints += "~{";
    constraints += clobber;
    constraints += '}';
  }

  // Add machine specific clobbers
  std::string_view machineClobbers = cgf.getTarget().getClobbers();
  if (!machineClobbers.empty()) {
    if (!constraints.empty())
      constraints += ',';
    constraints += machineClobbers;
  }
}

static void
collectInOutConstraintInfos(const CIRGenFunction &cgf, const AsmStmt &s,
                            SmallVectorImpl<TargetInfo::ConstraintInfo> &out,
                            SmallVectorImpl<TargetInfo::ConstraintInfo> &in) {

  for (unsigned i = 0, e = s.getNumOutputs(); i != e; ++i) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getOutputName(i);
    TargetInfo::ConstraintInfo info(s.getOutputConstraint(i), name);
    // `validateOutputConstraint` modifies the `info` object by setting the
    // read/write, clobber, allows-register, and allows-memory process.
    bool isValid = cgf.getTarget().validateOutputConstraint(info);
    (void)isValid;
    assert(isValid && "Failed to parse output constraint");
    out.push_back(info);
  }

  for (unsigned i = 0, e = s.getNumInputs(); i != e; ++i) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getInputName(i);
    TargetInfo::ConstraintInfo info(s.getInputConstraint(i), name);
    // `validateInputConstraint` modifies the `info` object by setting the
    // read/write, clobber, allows-register, and allows-memory process.
    bool isValid = cgf.getTarget().validateInputConstraint(out, info);
    assert(isValid && "Failed to parse input constraint");
    (void)isValid;
    in.push_back(info);
  }
}

std::pair<mlir::Value, mlir::Type> CIRGenFunction::emitAsmInputLValue(
    const TargetInfo::ConstraintInfo &info, LValue inputValue,
    QualType inputType, std::string &constraintString, SourceLocation loc) {

  if (info.allowsRegister() || !info.allowsMemory()) {
    if (hasScalarEvaluationKind(inputType))
      return {emitLoadOfLValue(inputValue, loc).getValue(), mlir::Type()};

    mlir::Type ty = convertType(inputType);
    uint64_t size = cgm.getDataLayout().getTypeSizeInBits(ty);
    if ((size <= 64 && llvm::isPowerOf2_64(size)) ||
        getTargetHooks().isScalarizableAsmOperand(*this, ty)) {
      ty = cir::IntType::get(&getMLIRContext(), size, false);

      return {builder.createLoad(
                  getLoc(loc),
                  inputValue.getAddress().withElementType(builder, ty)),
              mlir::Type()};
    }
  }

  Address addr = inputValue.getAddress();
  constraintString += '*';
  return {addr.getPointer(), addr.getElementType()};
}

std::pair<mlir::Value, mlir::Type>
CIRGenFunction::emitAsmInput(const TargetInfo::ConstraintInfo &info,
                             const Expr *inputExpr,
                             std::string &constraintString) {
  mlir::Location loc = getLoc(inputExpr->getExprLoc());

  // If this can't be a register or memory, i.e., has to be a constant
  // (immediate or symbolic), try to emit it as such.
  if (!info.allowsRegister() && !info.allowsMemory()) {
    if (info.requiresImmediateConstant()) {
      Expr::EvalResult evalResult;
      inputExpr->EvaluateAsRValue(evalResult, getContext(), true);

      llvm::APSInt intResult;
      if (evalResult.Val.toIntegralConstant(intResult, inputExpr->getType(),
                                            getContext()))
        return {builder.getConstInt(loc, intResult), mlir::Type()};
    }

    Expr::EvalResult result;
    if (inputExpr->EvaluateAsInt(result, getContext()))
      return {builder.getConstInt(loc, result.Val.getInt()), mlir::Type()};
  }

  if (info.allowsRegister() || !info.allowsMemory())
    if (CIRGenFunction::hasScalarEvaluationKind(inputExpr->getType()))
      return {emitScalarExpr(inputExpr), mlir::Type()};
  if (inputExpr->getStmtClass() == Expr::CXXThisExprClass)
    return {emitScalarExpr(inputExpr), mlir::Type()};
  inputExpr = inputExpr->IgnoreParenNoopCasts(getContext());
  LValue dest = emitLValue(inputExpr);
  return emitAsmInputLValue(info, dest, inputExpr->getType(), constraintString,
                            inputExpr->getExprLoc());
}

static void emitAsmStores(CIRGenFunction &cgf, const AsmStmt &s,
                          const llvm::ArrayRef<mlir::Value> regResults,
                          const llvm::ArrayRef<mlir::Type> resultRegTypes,
                          const llvm::ArrayRef<mlir::Type> resultTruncRegTypes,
                          const llvm::ArrayRef<LValue> resultRegDests,
                          const llvm::ArrayRef<QualType> resultRegQualTys,
                          const llvm::BitVector &resultTypeRequiresCast,
                          const llvm::BitVector &resultRegIsFlagReg) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.cgm;
  mlir::MLIRContext *ctx = builder.getContext();

  assert(regResults.size() == resultRegTypes.size());
  assert(regResults.size() == resultTruncRegTypes.size());
  assert(regResults.size() == resultRegDests.size());

  // ResultRegDests can be also populated by addReturnRegisterOutputs() above,
  // in which case its size may grow.
  assert(resultTypeRequiresCast.size() <= resultRegDests.size());
  assert(resultRegIsFlagReg.size() <= resultRegDests.size());

  for (unsigned i = 0, e = regResults.size(); i != e; ++i) {
    mlir::Value tmp = regResults[i];
    mlir::Type truncTy = resultTruncRegTypes[i];

    if (i < resultRegIsFlagReg.size() && resultRegIsFlagReg[i])
      assert(!cir::MissingFeatures::asmLLVMAssume());

    // If the result type of the LLVM IR asm doesn't match the result type of
    // the expression, do the conversion.
    if (resultRegTypes[i] != truncTy) {

      // Truncate the integer result to the right size, note that TruncTy can be
      // a pointer.
      if (mlir::isa<cir::FPTypeInterface>(truncTy)) {
        tmp = builder.createFloatingCast(tmp, truncTy);
      } else if (isa<cir::PointerType>(truncTy) &&
                 isa<cir::IntType>(tmp.getType())) {
        uint64_t resSize = cgm.getDataLayout().getTypeSizeInBits(truncTy);
        tmp = builder.createIntCast(
            tmp, cir::IntType::get(ctx, (unsigned)resSize, false));
        tmp = builder.createIntToPtr(tmp, truncTy);
      } else if (isa<cir::PointerType>(tmp.getType()) &&
                 isa<cir::IntType>(truncTy)) {
        uint64_t tmpSize = cgm.getDataLayout().getTypeSizeInBits(tmp.getType());
        tmp = builder.createPtrToInt(
            tmp, cir::IntType::get(ctx, (unsigned)tmpSize, false));
        tmp = builder.createIntCast(tmp, truncTy);
      } else if (isa<cir::IntType>(truncTy)) {
        tmp = builder.createIntCast(tmp, truncTy);
      } else if (isa<cir::VectorType>(truncTy)) {
        assert(!cir::MissingFeatures::asmVectorType());
      }
    }

    LValue dest = resultRegDests[i];
    // ResultTypeRequiresCast elements correspond to the first
    // ResultTypeRequiresCast.size() elements of RegResults.
    if ((i < resultTypeRequiresCast.size()) && resultTypeRequiresCast[i]) {
      unsigned size = cgf.getContext().getTypeSize(resultRegQualTys[i]);
      Address addr =
          dest.getAddress().withElementType(builder, resultRegTypes[i]);
      if (cgm.getTargetCIRGenInfo().isScalarizableAsmOperand(cgf, truncTy)) {
        builder.createStore(cgf.getLoc(s.getAsmLoc()), tmp, addr);
        continue;
      }

      QualType ty =
          cgf.getContext().getIntTypeForBitwidth(size, /*Signed=*/false);
      if (ty.isNull()) {
        const Expr *outExpr = s.getOutputExpr(i);
        cgm.getDiags().Report(outExpr->getExprLoc(),
                              diag::err_store_value_to_reg);
        return;
      }
      dest = cgf.makeAddrLValue(addr, ty);
    }

    cgf.emitStoreThroughLValue(RValue::get(tmp), dest);
  }
}

mlir::LogicalResult CIRGenFunction::emitAsmStmt(const AsmStmt &s) {
  // Assemble the final asm string.
  std::string asmString = s.generateAsmString(getContext());
  SourceLocation srcLoc = s.getAsmLoc();
  mlir::Location loc = getLoc(srcLoc);

  // Get all the output and input constraints together.
  SmallVector<TargetInfo::ConstraintInfo> outputConstraintInfos;
  SmallVector<TargetInfo::ConstraintInfo> inputConstraintInfos;
  collectInOutConstraintInfos(*this, s, outputConstraintInfos,
                              inputConstraintInfos);

  bool isGCCAsmGoto = false;

  std::string constraints;
  SmallVector<LValue> resultRegDests;
  SmallVector<QualType> resultRegQualTys;
  SmallVector<mlir::Type> resultRegTypes;
  SmallVector<mlir::Type> resultTruncRegTypes;
  SmallVector<mlir::Type> argTypes;
  SmallVector<mlir::Type> argElemTypes;
  SmallVector<mlir::Value> args;
  SmallVector<mlir::Value> outArgs;
  SmallVector<mlir::Value> inArgs;
  SmallVector<mlir::Value> inOutArgs;
  llvm::BitVector resultTypeRequiresCast;
  llvm::BitVector resultRegIsFlagReg;

  // Keep track of input constraints.
  std::string inOutConstraints;
  SmallVector<mlir::Type> inOutArgTypes;
  SmallVector<mlir::Type> inOutArgElemTypes;

  // Keep track of out constraints for tied input operand.
  SmallVector<std::string> outputConstraints;

  // Keep track of defined physregs.
  llvm::SmallSet<std::string, 8> physRegOutputs;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool readOnly = true, readNone = true;

  std::string outputConstraint;
  for (unsigned i = 0, e = s.getNumOutputs(); i != e; ++i) {
    TargetInfo::ConstraintInfo &info = outputConstraintInfos[i];

    // Simplify the output constraint.
    outputConstraint = s.getOutputConstraint(i);
    outputConstraint = getTarget().simplifyConstraint(
        StringRef(outputConstraint).drop_front());

    const Expr *outExpr = s.getOutputExpr(i);
    outExpr = outExpr->IgnoreParenNoopCasts(getContext());

    std::string gccReg;
    outputConstraint = s.addVariableConstraints(
        outputConstraint, *outExpr, getTarget(), info.earlyClobber(),
        [&](const Stmt *unspStmt, StringRef msg) {
          cgm.errorUnsupported(unspStmt, msg);
        },
        &gccReg);

    // Give an error on multiple outputs to same physreg.
    if (!gccReg.empty() && !physRegOutputs.insert(gccReg).second)
      cgm.error(srcLoc, "multiple outputs to hard register: " + gccReg);

    outputConstraints.push_back(outputConstraint);
    LValue dest = emitLValue(outExpr);

    if (!constraints.empty())
      constraints += ',';

    // If this is a register output, then make the inline a sm return it
    // by-value.  If this is a memory result, return the value by-reference.
    QualType qty = outExpr->getType();
    const bool isScalarOrAggregate =
        hasScalarEvaluationKind(qty) || hasAggregateEvaluationKind(qty);
    if (!info.allowsMemory() && isScalarOrAggregate) {
      constraints += "=" + outputConstraint;
      resultRegQualTys.push_back(qty);
      resultRegDests.push_back(dest);

      bool isFlagReg = llvm::StringRef(outputConstraint).starts_with("{@cc");
      resultRegIsFlagReg.push_back(isFlagReg);

      mlir::Type ty = convertTypeForMem(qty);
      const bool requiresCast =
          info.allowsRegister() &&
          (cgm.getTargetCIRGenInfo().isScalarizableAsmOperand(*this, ty) ||
           isa<cir::RecordType, cir::ArrayType>(ty));

      resultTruncRegTypes.push_back(ty);
      resultTypeRequiresCast.push_back(requiresCast);

      if (requiresCast) {
        unsigned size = getContext().getTypeSize(qty);
        if (size == 0)
          cgm.error(outExpr->getExprLoc(), "output size should not be zero");

        ty = cir::IntType::get(&getMLIRContext(), size, false);
      }

      resultRegTypes.push_back(ty);
      // If this output is tied to an input, and if the input is larger, then
      // we need to set the actual result type of the inline asm node to be the
      // same as the input type.
      if (info.hasMatchingInput()) {
        unsigned inputNo;
        for (inputNo = 0; inputNo != s.getNumInputs(); ++inputNo) {
          TargetInfo::ConstraintInfo &input = inputConstraintInfos[inputNo];
          if (input.hasTiedOperand() && input.getTiedOperand() == i)
            break;
        }
        assert(inputNo != s.getNumInputs() && "Didn't find matching input!");

        QualType inputTy = s.getInputExpr(inputNo)->getType();
        QualType outputType = outExpr->getType();

        uint64_t inputSize = getContext().getTypeSize(inputTy);
        if (getContext().getTypeSize(outputType) < inputSize) {
          // Form the asm to return the value as a larger integer or fp type.
          resultRegTypes.back() = convertType(inputTy);
        }

        if (mlir::Type adjTy = cgm.getTargetCIRGenInfo().adjustInlineAsmType(
                *this, outputConstraint, resultRegTypes.back()))
          resultRegTypes.back() = adjTy;
        else
          cgm.getDiags().Report(srcLoc, diag::err_asm_invalid_type_in_input)
              << outExpr->getType() << outputConstraint;
      }

      // Update largest vector width for any vector types.
      assert(!cir::MissingFeatures::asmVectorType());
    } else {
      Address destAddr = dest.getAddress();

      // Matrix types in memory are represented by arrays, but accessed through
      // vector pointers, with the alignment specified on the access operation.
      // For inline assembly, update pointer arguments to use vector pointers.
      // Otherwise there will be a mis-match if the matrix is also an
      // input-argument which is represented as vector.
      if (isa<MatrixType>(outExpr->getType().getCanonicalType()))
        destAddr =
            destAddr.withElementType(builder, convertType(outExpr->getType()));

      argTypes.push_back(destAddr.getType());
      argElemTypes.push_back(destAddr.getElementType());
      outArgs.push_back(destAddr.getPointer());
      args.push_back(destAddr.getPointer());
      constraints += "=*";
      constraints += outputConstraint;
      readOnly = readNone = false;
    }

    if (info.isReadWrite()) {
      inOutConstraints += ',';
      const Expr *inputExpr = s.getOutputExpr(i);

      // argValue: mlir::Value, argElementType: mlir::Type.
      auto [argValue, argElementType] =
          emitAsmInputLValue(info, dest, inputExpr->getType(), inOutConstraints,
                             inputExpr->getExprLoc());

      if (mlir::Type adjTy = getTargetHooks().adjustInlineAsmType(
              *this, outputConstraint, argValue.getType()))
        argValue = builder.createBitcast(argValue, adjTy);

      // Update largest vector width for any vector types.
      assert(!cir::MissingFeatures::asmVectorType());

      // Only tie earlyclobber physregs.
      if (info.allowsRegister() && (gccReg.empty() || info.earlyClobber()))
        inOutConstraints += llvm::utostr(i);
      else
        inOutConstraints += outputConstraint;

      inOutArgTypes.push_back(argValue.getType());
      inOutArgElemTypes.push_back(argElementType);
      inOutArgs.push_back(argValue);
    }

  } // iterate over output operands

  for (unsigned i = 0, e = s.getNumInputs(); i != e; ++i) {
    TargetInfo::ConstraintInfo &info = inputConstraintInfos[i];
    const Expr *inputExpr = s.getInputExpr(i);

    if (info.allowsMemory())
      readNone = false;

    if (!constraints.empty())
      constraints += ',';

    std::string inputConstraint(s.getInputConstraint(i));
    inputConstraint =
        getTarget().simplifyConstraint(inputConstraint, &outputConstraintInfos);

    inputConstraint = s.addVariableConstraints(
        inputConstraint, *inputExpr->IgnoreParenNoopCasts(getContext()),
        getTarget(), /*EarlyClobber=*/false,
        [&](const Stmt *unspStmt, StringRef msg) {
          cgm.errorUnsupported(unspStmt, msg);
        });

    std::string replaceConstraint(inputConstraint);
    // argValue: mlir::Value, argElementType: mlir::Type.
    auto [argValue, argElemType] = emitAsmInput(info, inputExpr, constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (info.hasTiedOperand()) {
      unsigned output = info.getTiedOperand();
      QualType outputType = s.getOutputExpr(output)->getType();
      QualType inputTy = inputExpr->getType();

      if (getContext().getTypeSize(outputType) >
          getContext().getTypeSize(inputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<cir::PointerType>(argValue.getType()))
          argValue = builder.createPtrToInt(argValue, uIntPtrTy);
        mlir::Type outputTy = convertType(outputType);
        if (isa<cir::IntType>(outputTy))
          argValue = builder.createIntCast(argValue, outputTy);
        else if (isa<cir::PointerType>(outputTy))
          argValue = builder.createIntCast(argValue, uIntPtrTy);
        else if (isa<cir::FPTypeInterface>(outputTy))
          argValue = builder.createFloatingCast(argValue, outputTy);
      }

      // Deal with the tied operands' constraint code in adjustInlineAsmType.
      replaceConstraint = outputConstraints[output];
    }

    if (mlir::Type adjTy = getTargetHooks().adjustInlineAsmType(
            *this, replaceConstraint, argValue.getType()))
      argValue = builder.createBitcast(argValue, adjTy);
    else
      cgm.getDiags().Report(s.getAsmLoc(), diag::err_asm_invalid_type_in_input)
          << inputExpr->getType() << inputConstraint;

    // Update largest vector width for any vector types.
    assert(!cir::MissingFeatures::asmVectorType());

    argTypes.push_back(argValue.getType());
    argElemTypes.push_back(argElemType);
    inArgs.push_back(argValue);
    args.push_back(argValue);
    constraints += inputConstraint;
  } // iterate over input operands

  // Append the "input" part of inout constraints.
  for (unsigned i = 0, e = inOutArgs.size(); i != e; ++i) {
    args.push_back(inOutArgs[i]);
    argTypes.push_back(inOutArgTypes[i]);
    argElemTypes.push_back(inOutArgElemTypes[i]);
  }
  constraints += inOutConstraints;

  bool hasUnwindClobber = false;
  collectClobbers(*this, s, constraints, hasUnwindClobber, readOnly, readNone);

  std::array<mlir::ValueRange, 3> operands = {outArgs, inArgs, inOutArgs};

  mlir::Type resultType;

  if (resultRegTypes.size() == 1)
    resultType = resultRegTypes[0];
  else if (resultRegTypes.size() > 1)
    resultType = builder.getAnonRecordTy(resultRegTypes, /*packed=*/false,
                                         /*padded=*/false);

  bool hasSideEffect = s.isVolatile() || s.getNumOutputs() == 0;

  std::vector<mlir::Value> regResults;
  cir::InlineAsmOp ia = cir::InlineAsmOp::create(
      builder, getLoc(s.getAsmLoc()), resultType, operands, asmString,
      constraints, hasSideEffect, inferFlavor(cgm, s), mlir::ArrayAttr());

  if (isGCCAsmGoto) {
    assert(!cir::MissingFeatures::asmGoto());
  } else if (hasUnwindClobber) {
    assert(!cir::MissingFeatures::asmUnwindClobber());
  } else {
    assert(!cir::MissingFeatures::asmMemoryEffects());

    mlir::Value result;
    if (ia.getNumResults())
      result = ia.getResult(0);

    llvm::SmallVector<mlir::Attribute> operandAttrs;

    int i = 0;
    for (auto typ : argElemTypes) {
      if (typ) {
        auto op = args[i++];
        assert(mlir::isa<cir::PointerType>(op.getType()) &&
               "pointer type expected");
        assert(cast<cir::PointerType>(op.getType()).getPointee() == typ &&
               "element type differs from pointee type!");

        operandAttrs.push_back(mlir::UnitAttr::get(&getMLIRContext()));
      } else {
        // We need to add an attribute for every arg since later, during
        // the lowering to LLVM IR the attributes will be assigned to the
        // CallInsn argument by index, i.e. we can't skip null type here
        operandAttrs.push_back(mlir::Attribute());
      }
    }
    assert(args.size() == operandAttrs.size() &&
           "The number of attributes is not even with the number of operands");

    ia.setOperandAttrsAttr(builder.getArrayAttr(operandAttrs));

    if (resultRegTypes.size() == 1) {
      regResults.push_back(result);
    } else if (resultRegTypes.size() > 1) {
      CharUnits alignment = CharUnits::One();
      mlir::Value dest =
          emitAlloca("__asm_result", resultType, loc, alignment, false);
      Address addr = Address(dest, alignment);
      builder.createStore(loc, result, addr);

      for (unsigned i = 0, e = resultRegTypes.size(); i != e; ++i) {
        cir::PointerType typ = builder.getPointerTo(resultRegTypes[i]);
        cir::GetMemberOp ptr = builder.createGetMember(loc, typ, dest, "", i);
        cir::LoadOp tmp = builder.createLoad(loc, Address(ptr, alignment));
        regResults.push_back(tmp);
      }
    }
  }

  emitAsmStores(*this, s, regResults, resultRegTypes, resultTruncRegTypes,
                resultRegDests, resultRegQualTys, resultTypeRequiresCast,
                resultRegIsFlagReg);

  return mlir::success();
}
