//=== MissingTerminatingZeroChecker.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check for string arguments passed to C library functions where the
// terminating zero is missing.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "llvm/ADT/BitVector.h"
#include <sstream>

using namespace clang;
using namespace ento;

namespace {

struct StringData {
  const MemRegion *StrRegion;
  int64_t StrLength;
  unsigned int Offset;
  const llvm::BitVector *NonNullData;
};

class MissingTerminatingZeroChecker
    : public Checker<check::Bind, check::PreCall> {
public:
  void checkBind(SVal L, SVal V, const Stmt *S, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

  void initOptions(bool NoDefaultIgnore, StringRef IgnoreList);

private:
  const BugType BT{this, "Missing terminating zero"};

  using IgnoreEntry = std::pair<int, int>;
  /// Functions (identified by name only) to ignore.
  /// The entry stores a parameter index, or -1.
  llvm::StringMap<IgnoreEntry> FunctionsToIgnore = {
      {"stpncpy", {1, -1}}, {"strncat", {1, -1}}, {"strncmp", {0, 1}},
      {"strncpy", {1, -1}}, {"strndup", {0, -1}}, {"strnlen", {0, -1}},
  };

  bool checkArg(unsigned int ArgI, CheckerContext &C,
                const CallEvent &Call) const;
  bool getStringData(StringData &DataOut, ProgramStateRef State,
                     SValBuilder &SVB, const MemRegion *StrReg) const;
  ProgramStateRef setStringData(ProgramStateRef State, Loc L,
                                const llvm::BitVector &NonNullData) const;
  void reportBug(ExplodedNode *N, const Expr *E, CheckerContext &C,
                 const char Msg[]) const;
};

} // namespace

namespace llvm {
template <> struct FoldingSetTrait<llvm::BitVector> {
  static inline void Profile(llvm::BitVector X, FoldingSetNodeID &ID) {
    ID.AddInteger(X.size());
    for (unsigned int I = 0; I < X.size(); ++I)
      ID.AddBoolean(X[I]);
  }
};
} // end namespace llvm

// Contains for a "string" (character array) region if elements are known to be
// non-zero. The bit vector is indexed by the array element index and is true
// if the element is known to be non-zero. Size of the vector does not
// correspond to the extent of the memory region (can be smaller), the missing
// elements are considered to be false.
// (A value of 'false' means that the string element is zero or unknown.)
REGISTER_MAP_WITH_PROGRAMSTATE(NonNullnessData, const MemRegion *,
                               llvm::BitVector)

void MissingTerminatingZeroChecker::checkBind(SVal L, SVal V, const Stmt *S,
                                              CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const MemRegion *MR = L.getAsRegion();

  if (const ElementRegion *ER = dyn_cast_or_null<ElementRegion>(MR)) {
    // Assign value to the index of an array.
    // Check for applicable array type.
    QualType ElemType = ER->getValueType().getCanonicalType();
    if (!ElemType->isCharType())
      return;
    if (C.getASTContext().getTypeSizeInChars(ElemType).getQuantity() != 1)
      return;

    RegionRawOffset ROffset = ER->getAsArrayOffset();
    unsigned int Index = ROffset.getOffset().getQuantity();

    // If the checker has data about the value to bind, use this information.
    // Otherwise try to get it from the analyzer.
    auto GetKnownToBeNonNull = [this, State, &C](SVal V) -> ConditionTruthVal {
      StringData ExistingSData;
      if (getStringData(ExistingSData, State, C.getSValBuilder(),
                        V.getAsRegion()) &&
          ExistingSData.Offset < ExistingSData.NonNullData->size()) {
        return ExistingSData.NonNullData->test(ExistingSData.Offset);
      } else {
        return State->isNonNull(V);
      }
    };
    ConditionTruthVal VKnownToBeNonNull = GetKnownToBeNonNull(V);

    if (const llvm::BitVector *NNData =
            State->get<NonNullnessData>(ROffset.getRegion())) {
      // Update existing data.
      unsigned int NewSize =
          Index < NNData->size() ? NNData->size() : Index + 1;
      // Only extend the vector with 'true' value.
      if (NewSize > NNData->size() && !VKnownToBeNonNull.isConstrainedTrue())
        return;
      llvm::BitVector NNData1(NewSize);
      NNData1 |= *NNData;
      NNData1[Index] = VKnownToBeNonNull.isConstrainedTrue();
      State = State->set<NonNullnessData>(ROffset.getRegion(), NNData1);
    } else {
      // Only add new data if 'true' is found.
      if (!VKnownToBeNonNull.isConstrainedTrue())
        return;
      llvm::BitVector NNData1(Index + 1);
      NNData1[Index] = true;
      State = State->set<NonNullnessData>(ROffset.getRegion(), NNData1);
    }
  } else if (const TypedValueRegion *TR =
                 dyn_cast_or_null<TypedValueRegion>(MR)) {
    // Initialize a region with compound value from list or string literal.
    QualType Type = TR->getValueType().getCanonicalType();
    if (!Type->isArrayType())
      return;
    if (!Type->castAsArrayTypeUnsafe()->getElementType()->isCharType())
      return;

    if (auto CVal = V.getAs<nonloc::CompoundVal>()) {
      llvm::BitVector NNData;
      for (auto Val = CVal->begin(); Val != CVal->end(); ++Val)
        NNData.push_back(State->isNonNull(*Val).isConstrainedTrue());
      State = State->set<NonNullnessData>(MR, NNData);
    } else if (auto MRV = V.getAs<loc::MemRegionVal>()) {
      if (auto *StrReg = MRV->stripCasts()->getAs<StringRegion>()) {
        StringRef Str = StrReg->getStringLiteral()->getString();
        size_t StrL = Str.size();
        llvm::BitVector NNData(StrL + 1, true);
        for (unsigned int I = 0; I < StrL; ++I)
          if (Str[I] == 0)
            NNData.reset(I);
        NNData.reset(StrL);
        State = State->set<NonNullnessData>(MR, NNData);
      }
    }
  }
  C.addTransition(State);
}

void MissingTerminatingZeroChecker::checkPreCall(const CallEvent &Call,
                                                 CheckerContext &C) const {
  if (!Call.isInSystemHeader() || !Call.isGlobalCFunction())
    return;

  auto Ignore = [this](StringRef Name) -> IgnoreEntry {
    auto FIgnore = FunctionsToIgnore.find(Name);
    if (FIgnore != FunctionsToIgnore.end())
      return FIgnore->getValue();
    return IgnoreEntry{-1, -1};
  }(cast<NamedDecl>(Call.getDecl())->getNameAsString());

  for (const auto &[ArgI, ParmD] : enumerate(Call.parameters())) {
    QualType ArgTy = ParmD->getType();
    if (!ArgTy->isPointerType())
      continue;
    QualType ArgPointeeTy = ArgTy->getPointeeType();
    if (!ArgPointeeTy->isCharType() || !ArgPointeeTy.isConstQualified())
      continue;
    if (static_cast<int>(ArgI) == Ignore.first ||
        static_cast<int>(ArgI) == Ignore.second)
      continue;

    if (checkArg(ArgI, C, Call))
      return;
  }
}

bool MissingTerminatingZeroChecker::checkArg(unsigned int ArgI,
                                             CheckerContext &C,
                                             const CallEvent &Call) const {
  SVal StrArgVal = Call.getArgSVal(ArgI);

  StringData SData;
  if (!getStringData(SData, C.getState(), C.getSValBuilder(),
                     StrArgVal.getAsRegion()))
    return false;

  // Check if all elements in the string are known to be non-zero.
  // The stored bitvector can have a smaller size.
  if (SData.NonNullData->size() < SData.StrLength)
    return false;
  for (int64_t I = SData.Offset; I < SData.StrLength; ++I)
    if (!SData.NonNullData->test(I))
      return false;

  reportBug(C.getPredecessor(), Call.getArgExpr(ArgI), C,
            "String contains no terminating zero; At this place a "
            "null-terminated string is expected");
  return true;
}

bool MissingTerminatingZeroChecker::getStringData(
    StringData &DataOut, ProgramStateRef State, SValBuilder &SVB,
    const MemRegion *StrReg) const {
  if (!StrReg)
    return false;

  unsigned int Offset = 0;
  if (const auto *ElemReg = dyn_cast_or_null<ElementRegion>(StrReg)) {
    RegionRawOffset ROffset = ElemReg->getAsArrayOffset();
    StrReg = ROffset.getRegion();
    if (!StrReg)
      return false;
    Offset = ROffset.getOffset().getQuantity();
  }

  const llvm::BitVector *NNData = State->get<NonNullnessData>(StrReg);
  if (!NNData)
    return false;

  DefinedOrUnknownSVal Extent = getDynamicExtent(State, StrReg, SVB);
  if (Extent.isUnknown())
    return false;
  const llvm::APSInt *KnownExtent = SVB.getKnownValue(State, Extent);
  if (!KnownExtent)
    return false;

  DataOut.StrRegion = StrReg;
  DataOut.StrLength = KnownExtent->getExtValue();
  DataOut.Offset = Offset;
  DataOut.NonNullData = NNData;

  return true;
}

void MissingTerminatingZeroChecker::reportBug(ExplodedNode *N, const Expr *E,
                                              CheckerContext &C,
                                              const char Msg[]) const {
  auto R = std::make_unique<PathSensitiveBugReport>(BT, Msg, N);
  bugreporter::trackExpressionValue(N, E, *R);
  C.emitReport(std::move(R));
}

void MissingTerminatingZeroChecker::initOptions(bool NoDefaultIgnore,
                                                StringRef IgnoreList) {
  if (NoDefaultIgnore)
    FunctionsToIgnore.clear();
  std::istringstream IgnoreInput{std::string(IgnoreList)};
  std::array<char, 100> I;
  while (IgnoreInput.getline(&I[0], 100, ';')) {
    std::istringstream FunctionInput{std::string(&I[0])};
    std::string FName;
    int FArg;
    if (FunctionInput >> FName >> FArg) {
      IgnoreEntry &E =
          FunctionsToIgnore.insert({FName, {-1, -1}}).first->getValue();
      if (E.first == -1)
        E.first = FArg;
      else if (E.second == -1)
        E.second = FArg;
    }
  }
}

void ento::registerMissingTerminatingZeroChecker(CheckerManager &Mgr) {
  auto *Checker = Mgr.registerChecker<MissingTerminatingZeroChecker>();
  bool NoDefaultIgnore = Mgr.getAnalyzerOptions().getCheckerBooleanOption(
      Checker, "OmitDefaultIgnoreFunctions");
  StringRef IgnoreList = Mgr.getAnalyzerOptions().getCheckerStringOption(
      Checker, "IgnoreFunctionArgs");
  Checker->initOptions(NoDefaultIgnore, IgnoreList);
}

bool ento::shouldRegisterMissingTerminatingZeroChecker(
    const CheckerManager &Mgr) {
  // Check if the char type has size of 8 bits (to avoid indexing differences)?
  return true;
}
