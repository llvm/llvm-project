#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/StringExtras.h"

#include "CIRGenFunction.h"
#include "TargetInfo.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

static AsmFlavor inferFlavor(const CIRGenModule &cgm, const AsmStmt &S) {
  AsmFlavor GnuAsmFlavor =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmFlavor::x86_att
          : AsmFlavor::x86_intel;

  return isa<MSAsmStmt>(&S) ? AsmFlavor::x86_intel : GnuAsmFlavor;
}

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
static std::string SimplifyConstraint(
    const char *Constraint, const TargetInfo &Target,
    SmallVectorImpl<TargetInfo::ConstraintInfo> *OutCons = nullptr) {
  std::string Result;

  while (*Constraint) {
    switch (*Constraint) {
    default:
      Result += Target.convertConstraint(Constraint);
      break;
    // Ignore these
    case '*':
    case '?':
    case '!':
    case '=': // Will see this and the following in mult-alt constraints.
    case '+':
      break;
    case '#': // Ignore the rest of the constraint alternative.
      while (Constraint[1] && Constraint[1] != ',')
        Constraint++;
      break;
    case '&':
    case '%':
      Result += *Constraint;
      while (Constraint[1] && Constraint[1] == *Constraint)
        Constraint++;
      break;
    case ',':
      Result += "|";
      break;
    case 'g':
      Result += "imr";
      break;
    case '[': {
      assert(OutCons &&
             "Must pass output names to constraints with a symbolic name");
      unsigned Index;
      bool result = Target.resolveSymbolicName(Constraint, *OutCons, Index);
      assert(result && "Could not resolve symbolic name");
      (void)result;
      Result += llvm::utostr(Index);
      break;
    }
    }

    Constraint++;
  }

  return Result;
}

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
/// Look at AsmExpr and if it is a variable declared
/// as using a particular register add that as a constraint that will be used
/// in this asm stmt.
static std::string
AddVariableConstraints(const std::string &Constraint, const Expr &AsmExpr,
                       const TargetInfo &Target, CIRGenModule &CGM,
                       const AsmStmt &Stmt, const bool EarlyClobber,
                       std::string *GCCReg = nullptr) {
  const DeclRefExpr *AsmDeclRef = dyn_cast<DeclRefExpr>(&AsmExpr);
  if (!AsmDeclRef)
    return Constraint;
  const ValueDecl &Value = *AsmDeclRef->getDecl();
  const VarDecl *Variable = dyn_cast<VarDecl>(&Value);
  if (!Variable)
    return Constraint;
  if (Variable->getStorageClass() != SC_Register)
    return Constraint;
  AsmLabelAttr *Attr = Variable->getAttr<AsmLabelAttr>();
  if (!Attr)
    return Constraint;
  StringRef Register = Attr->getLabel();
  assert(Target.isValidGCCRegisterName(Register));
  // We're using validateOutputConstraint here because we only care if
  // this is a register constraint.
  TargetInfo::ConstraintInfo Info(Constraint, "");
  if (Target.validateOutputConstraint(Info) && !Info.allowsRegister()) {
    CGM.ErrorUnsupported(&Stmt, "__asm__");
    return Constraint;
  }
  // Canonicalize the register here before returning it.
  Register = Target.getNormalizedGCCRegisterName(Register);
  if (GCCReg != nullptr)
    *GCCReg = Register.str();
  return (EarlyClobber ? "&{" : "{") + Register.str() + "}";
}

static void collectClobbers(const CIRGenFunction &cgf, const AsmStmt &S,
                            std::string &constraints, bool &hasUnwindClobber,
                            bool &readOnly, bool readNone) {

  hasUnwindClobber = false;
  auto &cgm = cgf.getCIRGenModule();

  // Clobbers
  for (unsigned i = 0, e = S.getNumClobbers(); i != e; i++) {
    StringRef clobber = S.getClobber(i);
    if (clobber == "memory")
      readOnly = readNone = false;
    else if (clobber == "unwind") {
      hasUnwindClobber = true;
      continue;
    } else if (clobber != "cc") {
      clobber = cgf.getTarget().getNormalizedGCCRegisterName(clobber);
      if (cgm.getCodeGenOpts().StackClashProtector &&
          cgf.getTarget().isSPRegName(clobber)) {
        cgm.getDiags().Report(S.getAsmLoc(),
                              diag::warn_stack_clash_protection_inline_asm);
      }
    }

    if (isa<MSAsmStmt>(&S)) {
      if (clobber == "eax" || clobber == "edx") {
        if (constraints.find("=&A") != std::string::npos)
          continue;
        std::string::size_type position1 =
            constraints.find("={" + clobber.str() + "}");
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

using constraintInfos = SmallVector<TargetInfo::ConstraintInfo, 4>;

static void collectInOutConstrainsInfos(const CIRGenFunction &cgf,
                                        const AsmStmt &S, constraintInfos &out,
                                        constraintInfos &in) {

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getOutputName(i);
    TargetInfo::ConstraintInfo Info(S.getOutputConstraint(i), Name);
    bool IsValid = cgf.getTarget().validateOutputConstraint(Info);
    (void)IsValid;
    assert(IsValid && "Failed to parse output constraint");
    out.push_back(Info);
  }

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    StringRef Name;
    if (const GCCAsmStmt *GAS = dyn_cast<GCCAsmStmt>(&S))
      Name = GAS->getInputName(i);
    TargetInfo::ConstraintInfo Info(S.getInputConstraint(i), Name);
    bool IsValid = cgf.getTarget().validateInputConstraint(out, Info);
    assert(IsValid && "Failed to parse input constraint");
    (void)IsValid;
    in.push_back(Info);
  }
}

mlir::Value CIRGenFunction::buildAsmInputLValue(
    const TargetInfo::ConstraintInfo &Info, LValue InputValue,
    QualType InputType, std::string &ConstraintStr, SourceLocation Loc) {

  if (Info.allowsRegister() || !Info.allowsMemory()) {
    if (hasScalarEvaluationKind(InputType))
      return buildLoadOfLValue(InputValue, Loc).getScalarVal();

    mlir::Type Ty = convertType(InputType);
    uint64_t Size = CGM.getDataLayout().getTypeSizeInBits(Ty);
    if ((Size <= 64 && llvm::isPowerOf2_64(Size)) ||
        getTargetHooks().isScalarizableAsmOperand(*this, Ty)) {
      Ty = mlir::cir::IntType::get(builder.getContext(), Size, false);

      return builder.createLoad(getLoc(Loc),
                                InputValue.getAddress().withElementType(Ty));
    }
  }

  Address Addr = InputValue.getAddress();
  ConstraintStr += '*';
  return Addr.getPointer();
}

mlir::Value
CIRGenFunction::buildAsmInput(const TargetInfo::ConstraintInfo &Info,
                              const Expr *InputExpr,
                              std::string &ConstraintStr) {
  auto loc = getLoc(InputExpr->getExprLoc());

  // If this can't be a register or memory, i.e., has to be a constant
  // (immediate or symbolic), try to emit it as such.
  if (!Info.allowsRegister() && !Info.allowsMemory()) {
    if (Info.requiresImmediateConstant()) {
      Expr::EvalResult EVResult;
      InputExpr->EvaluateAsRValue(EVResult, getContext(), true);

      llvm::APSInt IntResult;
      if (EVResult.Val.toIntegralConstant(IntResult, InputExpr->getType(),
                                          getContext()))
        return builder.getConstAPSInt(loc, IntResult);
    }

    Expr::EvalResult Result;
    if (InputExpr->EvaluateAsInt(Result, getContext()))
      builder.getConstAPSInt(loc, Result.Val.getInt());
  }

  if (Info.allowsRegister() || !Info.allowsMemory())
    if (CIRGenFunction::hasScalarEvaluationKind(InputExpr->getType()))
      return buildScalarExpr(InputExpr);
  if (InputExpr->getStmtClass() == Expr::CXXThisExprClass)
    return buildScalarExpr(InputExpr);
  InputExpr = InputExpr->IgnoreParenNoopCasts(getContext());
  LValue Dest = buildLValue(InputExpr);
  return buildAsmInputLValue(Info, Dest, InputExpr->getType(), ConstraintStr,
                             InputExpr->getExprLoc());
}

mlir::LogicalResult CIRGenFunction::buildAsmStmt(const AsmStmt &S) {
  // Assemble the final asm string.
  std::string AsmString = S.generateAsmString(getContext());

  // Get all the output and input constraints together.
  constraintInfos OutputConstraintInfos;
  constraintInfos InputConstraintInfos;
  collectInOutConstrainsInfos(*this, S, OutputConstraintInfos,
                              InputConstraintInfos);

  std::string Constraints;
  std::vector<mlir::Type> ResultRegTypes;
  std::vector<mlir::Value> Args;

  // Keep track of input constraints.
  std::string InOutConstraints;
  std::vector<mlir::Value> InOutArgs;

  // Keep track of out constraints for tied input operand.
  std::vector<std::string> OutputConstraints;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool ReadOnly = true, ReadNone = true;

  for (unsigned i = 0, e = S.getNumOutputs(); i != e; i++) {
    TargetInfo::ConstraintInfo &Info = OutputConstraintInfos[i];

    // Simplify the output constraint.
    std::string OutputConstraint(S.getOutputConstraint(i));
    OutputConstraint = SimplifyConstraint(OutputConstraint.c_str() + 1,
                                          getTarget(), &OutputConstraintInfos);

    const Expr *OutExpr = S.getOutputExpr(i);
    OutExpr = OutExpr->IgnoreParenNoopCasts(getContext());

    std::string GCCReg;
    OutputConstraint =
        AddVariableConstraints(OutputConstraint, *OutExpr, getTarget(), CGM, S,
                               Info.earlyClobber(), &GCCReg);

    OutputConstraints.push_back(OutputConstraint);
    LValue Dest = buildLValue(OutExpr);

    if (!Constraints.empty())
      Constraints += ',';

    // If this is a register output, then make the inline a sm return it
    // by-value.  If this is a memory result, return the value by-reference.
    QualType QTy = OutExpr->getType();
    const bool IsScalarOrAggregate =
        hasScalarEvaluationKind(QTy) || hasAggregateEvaluationKind(QTy);
    if (!Info.allowsMemory() && IsScalarOrAggregate) {
      Constraints += "=" + OutputConstraint;
    } else {
      Address DestAddr = Dest.getAddress();

      // Matrix types in memory are represented by arrays, but accessed through
      // vector pointers, with the alignment specified on the access operation.
      // For inline assembly, update pointer arguments to use vector pointers.
      // Otherwise there will be a mis-match if the matrix is also an
      // input-argument which is represented as vector.
      if (isa<MatrixType>(OutExpr->getType().getCanonicalType()))
        DestAddr = DestAddr.withElementType(ConvertType(OutExpr->getType()));

      Args.push_back(DestAddr.getPointer());
      Constraints += "=*";
      Constraints += OutputConstraint;
    }

    if (Info.isReadWrite()) {
      InOutConstraints += ',';
      const Expr *InputExpr = S.getOutputExpr(i);

      mlir::Value Arg =
          buildAsmInputLValue(Info, Dest, InputExpr->getType(),
                              InOutConstraints, InputExpr->getExprLoc());

      if (mlir::Type AdjTy = getTargetHooks().adjustInlineAsmType(
              *this, OutputConstraint, Arg.getType()))
        Arg = builder.createBitcast(Arg, AdjTy);

      // Only tie earlyclobber physregs.
      if (Info.allowsRegister() && (GCCReg.empty() || Info.earlyClobber()))
        InOutConstraints += llvm::utostr(i);
      else
        InOutConstraints += OutputConstraint;

      InOutArgs.push_back(Arg);
    }
  } // iterate over output operands

  for (unsigned i = 0, e = S.getNumInputs(); i != e; i++) {
    const Expr *InputExpr = S.getInputExpr(i);

    TargetInfo::ConstraintInfo &Info = InputConstraintInfos[i];

    if (!Constraints.empty())
      Constraints += ',';

    // Simplify the input constraint.
    std::string InputConstraint(S.getInputConstraint(i));
    InputConstraint = SimplifyConstraint(InputConstraint.c_str(), getTarget(),
                                         &OutputConstraintInfos);

    InputConstraint = AddVariableConstraints(
        InputConstraint, *InputExpr->IgnoreParenNoopCasts(getContext()),
        getTarget(), CGM, S, false /* No EarlyClobber */);

    std::string ReplaceConstraint(InputConstraint);
    mlir::Value Arg = buildAsmInput(Info, InputExpr, Constraints);

    // If this input argument is tied to a larger output result, extend the
    // input to be the same size as the output.  The LLVM backend wants to see
    // the input and output of a matching constraint be the same size.  Note
    // that GCC does not define what the top bits are here.  We use zext because
    // that is usually cheaper, but LLVM IR should really get an anyext someday.
    if (Info.hasTiedOperand()) {
      unsigned Output = Info.getTiedOperand();
      QualType OutputType = S.getOutputExpr(Output)->getType();
      QualType InputTy = InputExpr->getType();

      if (getContext().getTypeSize(OutputType) >
          getContext().getTypeSize(InputTy)) {
        // Use ptrtoint as appropriate so that we can do our extension.
        if (isa<mlir::cir::PointerType>(Arg.getType()))
          Arg = builder.createPtrToInt(Arg, UIntPtrTy);
        mlir::Type OutputTy = convertType(OutputType);
        if (isa<mlir::cir::IntType>(OutputTy))
          Arg = builder.createIntCast(Arg, OutputTy);
        else if (isa<mlir::cir::PointerType>(OutputTy))
          Arg = builder.createIntCast(Arg, UIntPtrTy);
        else if (isa<mlir::FloatType>(OutputTy))
          Arg = builder.createFloatingCast(Arg, OutputTy);
      }

      // Deal with the tied operands' constraint code in adjustInlineAsmType.
      ReplaceConstraint = OutputConstraints[Output];
    }

    if (mlir::Type AdjTy = getTargetHooks().adjustInlineAsmType(
            *this, ReplaceConstraint, Arg.getType()))
      Arg = builder.createBitcast(Arg, AdjTy);
    else
      CGM.getDiags().Report(S.getAsmLoc(), diag::err_asm_invalid_type_in_input)
          << InputExpr->getType() << InputConstraint;

    Args.push_back(Arg);
    Constraints += InputConstraint;
  } // iterate over input operands

  // Append the "input" part of inout constraints.
  for (unsigned i = 0, e = InOutArgs.size(); i != e; i++) {
    Args.push_back(InOutArgs[i]);
  }
  Constraints += InOutConstraints;

  bool HasUnwindClobber = false;
  collectClobbers(*this, S, Constraints, HasUnwindClobber, ReadOnly, ReadNone);

  mlir::Type ResultType;

  if (ResultRegTypes.size() == 1)
    ResultType = ResultRegTypes[0];
  else if (ResultRegTypes.size() > 1) {
    auto sname = builder.getUniqueAnonRecordName();
    ResultType =
        builder.getCompleteStructTy(ResultRegTypes, sname, false, nullptr);
  }

  AsmFlavor AsmFlavor = inferFlavor(CGM, S);

  builder.create<mlir::cir::InlineAsmOp>(getLoc(S.getAsmLoc()), ResultType,
                                         Args, AsmString, Constraints,
                                         AsmFlavor);

  return mlir::success();
}