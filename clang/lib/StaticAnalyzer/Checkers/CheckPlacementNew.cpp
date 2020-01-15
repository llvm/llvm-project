#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang;
using namespace ento;

namespace {
class PlacementNewChecker : public Checker<check::PreStmt<CXXNewExpr>> {
public:
  void checkPreStmt(const CXXNewExpr *NE, CheckerContext &C) const;

private:
  // Returns the size of the target in a placement new expression.
  // E.g. in "new (&s) long" it returns the size of `long`.
  SVal getExtentSizeOfNewTarget(const CXXNewExpr *NE, ProgramStateRef State,
                                CheckerContext &C) const;
  // Returns the size of the place in a placement new expression.
  // E.g. in "new (&s) long" it returns the size of `s`.
  SVal getExtentSizeOfPlace(const Expr *NE, ProgramStateRef State,
                            CheckerContext &C) const;
  BugType BT{this, "Insufficient storage for placement new",
             categories::MemoryError};
};
} // namespace

SVal PlacementNewChecker::getExtentSizeOfPlace(const Expr *Place,
                                               ProgramStateRef State,
                                               CheckerContext &C) const {
  const MemRegion *MRegion = C.getSVal(Place).getAsRegion();
  if (!MRegion)
    return UnknownVal();
  RegionOffset Offset = MRegion->getAsOffset();
  if (Offset.hasSymbolicOffset())
    return UnknownVal();
  const MemRegion *BaseRegion = MRegion->getBaseRegion();
  if (!BaseRegion)
    return UnknownVal();

  SValBuilder &SvalBuilder = C.getSValBuilder();
  NonLoc OffsetInBytes = SvalBuilder.makeArrayIndex(
      Offset.getOffset() / C.getASTContext().getCharWidth());
  DefinedOrUnknownSVal ExtentInBytes =
      BaseRegion->castAs<SubRegion>()->getExtent(SvalBuilder);

  return SvalBuilder.evalBinOp(State, BinaryOperator::Opcode::BO_Sub,
                               ExtentInBytes, OffsetInBytes,
                               SvalBuilder.getArrayIndexType());
}

SVal PlacementNewChecker::getExtentSizeOfNewTarget(const CXXNewExpr *NE,
                                                   ProgramStateRef State,
                                                   CheckerContext &C) const {
  SValBuilder &SvalBuilder = C.getSValBuilder();
  QualType ElementType = NE->getAllocatedType();
  ASTContext &AstContext = C.getASTContext();
  CharUnits TypeSize = AstContext.getTypeSizeInChars(ElementType);
  if (NE->isArray()) {
    const Expr *SizeExpr = *NE->getArraySize();
    SVal ElementCount = C.getSVal(SizeExpr);
    if (auto ElementCountNL = ElementCount.getAs<NonLoc>()) {
      // size in Bytes = ElementCountNL * TypeSize
      return SvalBuilder.evalBinOp(
          State, BO_Mul, *ElementCountNL,
          SvalBuilder.makeArrayIndex(TypeSize.getQuantity()),
          SvalBuilder.getArrayIndexType());
    }
  } else {
    // Create a concrete int whose size in bits and signedness is equal to
    // ArrayIndexType.
    llvm::APInt I(AstContext.getTypeSizeInChars(SvalBuilder.getArrayIndexType())
                          .getQuantity() *
                      C.getASTContext().getCharWidth(),
                  TypeSize.getQuantity());
    return SvalBuilder.makeArrayIndex(I.getZExtValue());
  }
  return UnknownVal();
}

void PlacementNewChecker::checkPreStmt(const CXXNewExpr *NE,
                                       CheckerContext &C) const {
  // Check only the default placement new.
  if (!NE->getOperatorNew()->isReservedGlobalPlacementOperator())
    return;
  if (NE->getNumPlacementArgs() == 0)
    return;

  ProgramStateRef State = C.getState();
  SVal SizeOfTarget = getExtentSizeOfNewTarget(NE, State, C);
  const Expr *Place = NE->getPlacementArg(0);
  SVal SizeOfPlace = getExtentSizeOfPlace(Place, State, C);
  const auto SizeOfTargetCI = SizeOfTarget.getAs<nonloc::ConcreteInt>();
  if (!SizeOfTargetCI)
    return;
  const auto SizeOfPlaceCI = SizeOfPlace.getAs<nonloc::ConcreteInt>();
  if (!SizeOfPlaceCI)
    return;

  if (SizeOfPlaceCI->getValue() < SizeOfTargetCI->getValue()) {
    if (ExplodedNode *N = C.generateErrorNode(State)) {
      std::string Msg =
          llvm::formatv("Storage provided to placement new is only {0} bytes, "
                        "whereas the allocated type requires {1} bytes",
                        SizeOfPlaceCI->getValue(), SizeOfTargetCI->getValue());

      auto R = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
      bugreporter::trackExpressionValue(N, Place, *R);
      C.emitReport(std::move(R));
      return;
    }
  }
}

void ento::registerPlacementNewChecker(CheckerManager &mgr) {
  mgr.registerChecker<PlacementNewChecker>();
}

bool ento::shouldRegisterPlacementNewChecker(const LangOptions &LO) {
  return true;
}
