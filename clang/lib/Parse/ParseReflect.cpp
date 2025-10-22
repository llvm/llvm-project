



#include "clang/AST/LocInfoType.h"
#include "clang/Basic/DiagnosticParse.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Sema/EnterExpressionEvaluationContext.h"
using namespace clang;

ExprResult Parser::ParseCXXReflectExpression(SourceLocation OpLoc) {
  assert(Tok.is(tok::caretcaret) && "expected '^^'");
  EnterExpressionEvaluationContext Unevaluated(
      Actions, Sema::ExpressionEvaluationContext::Unevaluated);

  SourceLocation OperandLoc = Tok.getLocation();

  // Parse a leading nested-name-specifier
  CXXScopeSpec SS;
  if (ParseOptionalCXXScopeSpecifier(SS, /*ObjectType=*/nullptr,
                                     /*ObjectHasErrors=*/false,
                                     /*EnteringContext=*/false)) {
    SkipUntil(tok::semi, StopAtSemi | StopBeforeMatch);
    return ExprError();
  }

  {
    TentativeParsingAction TPA(*this);

    if (SS.isValid() &&
               SS.getScopeRep().getKind() == NestedNameSpecifier::Kind::Global) {
      // Check for global namespace '^^::'
      TPA.Commit();
      Decl *TUDecl = Actions.getASTContext().getTranslationUnitDecl();
      return Actions.ActOnCXXReflectExpr(OpLoc, SourceLocation(), TUDecl);
    }
    TPA.Revert();
  }

  if (isCXXTypeId(TentativeCXXTypeIdContext::AsReflectionOperand)) {
    TypeResult TR = ParseTypeName(/*TypeOf=*/nullptr);
    if (TR.isInvalid())
      return ExprError();

    TypeSourceInfo *TSI = nullptr;
    QualType QT = Actions.GetTypeFromParser(TR.get(), &TSI);

    return Actions.ActOnCXXReflectExpr(OpLoc, TSI);
  }

  Diag(OperandLoc, diag::err_cannot_reflect_operand);
  return ExprError();
}
