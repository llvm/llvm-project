#include "clang/Basic/DiagnosticSema.h"
#include "llvm/ADT/StringExtras.h"

#include "CIRGenFunction.h"
#include "TargetInfo.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

static AsmDialect inferDialect(const CIRGenModule &cgm, const AsmStmt &S) {
  AsmDialect GnuAsmDialect =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmDialect::x86_att
          : AsmDialect::x86_intel;

  return isa<MSAsmStmt>(&S) ? AsmDialect::x86_intel : GnuAsmDialect;
}

mlir::LogicalResult CIRGenFunction::buildAsmStmt(const AsmStmt &S) {
  // Assemble the final asm string.
  std::string AsmString = S.generateAsmString(getContext());

  std::string Constraints;
  std::vector<mlir::Type> ResultRegTypes;
  std::vector<mlir::Value> Args;

  assert(!S.getNumOutputs() && "asm output operands are NYI");
  assert(!S.getNumInputs() && "asm intput operands are NYI");
  assert(!S.getNumClobbers() && "asm clobbers operands are NYI");

  mlir::Type ResultType;

  if (ResultRegTypes.size() == 1)
    ResultType = ResultRegTypes[0];
  else if (ResultRegTypes.size() > 1) {
    auto sname = builder.getUniqueAnonRecordName();
    ResultType =
        builder.getCompleteStructTy(ResultRegTypes, sname, false, nullptr);
  }

  AsmDialect AsmDialect = inferDialect(CGM, S);

  builder.create<mlir::cir::InlineAsmOp>(getLoc(S.getAsmLoc()), ResultType,
                                         AsmString, AsmDialect);

  return mlir::success();
}