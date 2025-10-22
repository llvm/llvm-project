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

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
static std::string simplifyConstraint(
    const char *constraint, const TargetInfo &target,
    SmallVectorImpl<TargetInfo::ConstraintInfo> *outCons = nullptr) {
  std::string result;

  while (*constraint) {
    switch (*constraint) {
    default:
      result += target.convertConstraint(constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case '#': // Ignore the rest of the constraint alternative.
      while (constraint[1] && constraint[1] != ',')
        constraint++;
      break;
    case '&':
    case '%':
      result += *constraint;
      while (constraint[1] && constraint[1] == *constraint)
        constraint++;
      break;
    case ',':
      result += "|";
      break;
    case 'g':
      result += "imr";
      break;
    case '[': {
      assert(outCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned index;
      bool resolveResult =
          target.resolveSymbolicName(constraint, *outCons, index);
      assert(resolveResult && "Could not resolve symbolic name");
      (void)resolveResult;
      result += llvm::utostr(index);
      break;
    }
    }

    constraint++;
  }

  return result;
}

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
/// Look at AsmExpr and if it is a variable declared
/// as using a particular register add that as a constraint that will be used
/// in this asm stmt.
static std::string
addVariableConstraints(const std::string &constraint, const Expr &asmExpr,
                       const TargetInfo &target, CIRGenModule &cgm,
                       const AsmStmt &stmt, const bool earlyClobber,
                       std::string *gccReg = nullptr) {
  const DeclRefExpr *asmDeclRef = dyn_cast<DeclRefExpr>(&asmExpr);
  if (!asmDeclRef)
    return constraint;
  const ValueDecl &value = *asmDeclRef->getDecl();
  const VarDecl *variable = dyn_cast<VarDecl>(&value);
  if (!variable)
    return constraint;
  if (variable->getStorageClass() != SC_Register)
    return constraint;
  AsmLabelAttr *attr = variable->getAttr<AsmLabelAttr>();
  if (!attr)
    return constraint;
  StringRef registerName = attr->getLabel();
  assert(target.isValidGCCRegisterName(registerName));
  // We're using validateOutputConstraint here because we only care if
  // this is a register constraint.
  TargetInfo::ConstraintInfo info(constraint, "");
  if (target.validateOutputConstraint(info) && !info.allowsRegister()) {
    cgm.errorUnsupported(&stmt, "__asm__");
    return constraint;
  }
  // Canonicalize the register here before returning it.
  registerName = target.getNormalizedGCCRegisterName(registerName);
  if (gccReg != nullptr)
    *gccReg = registerName.str();
  return (earlyClobber ? "&{" : "{") + registerName.str() + "}";
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

using ConstraintInfos = SmallVector<TargetInfo::ConstraintInfo, 4>;

static void collectInOutConstraintInfos(const CIRGenFunction &cgf,
                                        const AsmStmt &s, ConstraintInfos &out,
                                        ConstraintInfos &in) {

  for (unsigned i = 0, e = s.getNumOutputs(); i != e; i++) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getOutputName(i);
    TargetInfo::ConstraintInfo info(s.getOutputConstraint(i), name);
    bool isValid = cgf.getTarget().validateOutputConstraint(info);
    (void)isValid;
    assert(isValid && "Failed to parse output constraint");
    out.push_back(info);
  }

  for (unsigned i = 0, e = s.getNumInputs(); i != e; i++) {
    StringRef name;
    if (const GCCAsmStmt *gas = dyn_cast<GCCAsmStmt>(&s))
      name = gas->getInputName(i);
    TargetInfo::ConstraintInfo info(s.getInputConstraint(i), name);
    bool isValid = cgf.getTarget().validateInputConstraint(out, info);
    assert(isValid && "Failed to parse input constraint");
    (void)isValid;
    in.push_back(info);
  }
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
      if (mlir::isa<mlir::FloatType>(truncTy)) {
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
  ConstraintInfos outputConstraintInfos;
  ConstraintInfos inputConstraintInfos;
  collectInOutConstraintInfos(*this, s, outputConstraintInfos,
                              inputConstraintInfos);

  bool isGCCAsmGoto = false;

  std::string constraints;
  std::vector<LValue> resultRegDests;
  std::vector<QualType> resultRegQualTys;
  std::vector<mlir::Type> resultRegTypes;
  std::vector<mlir::Type> resultTruncRegTypes;
  std::vector<mlir::Type> argTypes;
  std::vector<mlir::Type> argElemTypes;
  std::vector<mlir::Value> args;
  std::vector<mlir::Value> outArgs;
  std::vector<mlir::Value> inArgs;
  std::vector<mlir::Value> inOutArgs;
  llvm::BitVector resultTypeRequiresCast;
  llvm::BitVector resultRegIsFlagReg;

  // Keep track of out constraints for tied input operand.
  std::vector<std::string> outputConstraints;

  // Keep track of defined physregs.
  llvm::SmallSet<std::string, 8> physRegOutputs;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool readOnly = true, readNone = true;

  if (s.getNumInputs() != 0) {
    assert(!cir::MissingFeatures::asmInputOperands());
    cgm.errorNYI(srcLoc, "asm with input operands");
  }

  for (unsigned i = 0, e = s.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &info = outputConstraintInfos[i];

    // Simplify the output constraint.
    std::string outputConstraint(s.getOutputConstraint(i));
    outputConstraint =
        simplifyConstraint(outputConstraint.c_str() + 1, getTarget());

    const Expr *outExpr = s.getOutputExpr(i);
    outExpr = outExpr->IgnoreParenNoopCasts(getContext());

    std::string gccReg;
    outputConstraint =
        addVariableConstraints(outputConstraint, *outExpr, getTarget(), cgm, s,
                               info.earlyClobber(), &gccReg);

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
        ty = cir::IntType::get(&getMLIRContext(), size, false);
      }

      resultRegTypes.push_back(ty);

      if (info.hasMatchingInput())
        assert(!cir::MissingFeatures::asmInputOperands());

      if (mlir::Type adjTy = cgm.getTargetCIRGenInfo().adjustInlineAsmType(
              *this, outputConstraint, resultRegTypes.back()))
        resultRegTypes.back() = adjTy;
      else
        cgm.getDiags().Report(srcLoc, diag::err_asm_invalid_type_in_input)
            << outExpr->getType() << outputConstraint;

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

    if (info.isReadWrite())
      assert(!cir::MissingFeatures::asmInputOperands());

  } // iterate over output operands

  bool hasUnwindClobber = false;
  collectClobbers(*this, s, constraints, hasUnwindClobber, readOnly, readNone);

  std::array<mlir::ValueRange, 3> operands = {outArgs, inArgs, inOutArgs};

  mlir::Type resultType;

  if (resultRegTypes.size() == 1) {
    resultType = resultRegTypes[0];
  } else if (resultRegTypes.size() > 1) {
    std::string sname = builder.getUniqueAnonRecordName();
    resultType =
        builder.getCompleteRecordTy(resultRegTypes, sname, false, false);
  }
  bool hasSideEffect = s.isVolatile() || s.getNumOutputs() == 0;

  std::vector<mlir::Value> regResults;

  cir::InlineAsmOp ia = builder.create<cir::InlineAsmOp>(
      loc, resultType, operands, asmString, constraints, hasSideEffect,
      inferFlavor(cgm, s), mlir::ArrayAttr());

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
      mlir::StringAttr sname = cast<cir::RecordType>(resultType).getName();
      mlir::Value dest = emitAlloca(sname, resultType, loc, alignment, false);
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
