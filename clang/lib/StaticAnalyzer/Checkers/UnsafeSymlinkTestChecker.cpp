//===-- UnsafeSymlinkTestChecker.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker that checks for incorrect symlink detection.
// The checker works according to the rule CERT POS35-C. "Avoid race conditions
// while checking for the existence of a symbolic link".
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

/// Used to identify a file name.
/// If created with a symbolic region, use the region as key.
/// If created with a string region, use the contained string as key (different
/// string regions with same content should be equal).
class FileNameKey {
  std::string FileNameStr;
  const MemRegion *Region = nullptr;

public:
  FileNameKey(const MemRegion *R) {
    R = R->StripCasts();
    if (const auto *SR = dyn_cast<StringRegion>(R))
      FileNameStr = SR->getStringLiteral()->getString();
    else
      Region = R;
  }

  bool operator==(const FileNameKey &RHS) const {
    return std::tie(FileNameStr, Region) ==
           std::tie(RHS.FileNameStr, RHS.Region);
  }

  bool operator<(const FileNameKey &RHS) const {
    return std::tie(FileNameStr, Region) <
           std::tie(RHS.FileNameStr, RHS.Region);
  }

  const MemRegion *getRegionOrNull() const { return Region; }

  std::string getFileName(llvm::StringRef PrefixStr) const {
    if (!Region)
      return (llvm::Twine(PrefixStr) + "'" + FileNameStr + "'").str();
    return "";
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddString(FileNameStr);
    ID.AddPointer(Region);
  }
};

/// Data maintained about a region belonging to a "struct stat".
struct StatData {
  /// Region of a 'struct stat' object.
  const SubRegion *Region;
  /// Value of the field 'st_mode'.
  SVal StModeVal;
  /// Value of the field 'st_ino'.
  SVal StInoVal;
  /// Value of the field 'st_dev'.
  SVal StDevVal;

  bool operator==(const StatData &D) const {
    return std::tie(Region, StModeVal, StInoVal, StDevVal) ==
           std::tie(D.Region, D.StModeVal, D.StInoVal, D.StDevVal);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Region);
    ID.Add(StModeVal);
    ID.Add(StInoVal);
    ID.Add(StDevVal);
  }
};

/// Data about a file after `lstat` (but not `open`) was called.
struct FileDataLStat {
  /// Information about the `stat` structure that was passed to `lstat`.
  StatData LStatD;
  /// Indicates if a test for symbolic link on the `st_mode` field of the `stat`
  /// structure was performed, using the `S_ISLNK` macro.
  bool LinkCheckPerformed;

  bool operator==(const FileDataLStat &R) const {
    return std::tie(LStatD, LinkCheckPerformed) ==
           std::tie(R.LStatD, R.LinkCheckPerformed);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.Add(LStatD);
    ID.AddBoolean(LinkCheckPerformed);
  }
};

/// Data about a file after `lstat` and `open` was called.
struct FileDataOpened {
  /// Information about the `stat` structure that was passed to `lstat`.
  StatData LStatD;
  /// Information about the `stat` structure that was passed to `fstat`.
  StatData FStatD;
  /// Indicates if a test for symbolic link on the `st_mode` field of any of the
  /// `stat` structures was performed, using the `S_ISLNK` macro.
  bool LinkCheckPerformed;
  /// Data about the file name (this is used for checker messages).
  FileNameKey FName;

  bool operator==(const FileDataOpened &R) const {
    return std::tie(LStatD, FStatD, LinkCheckPerformed) ==
           std::tie(R.LStatD, R.FStatD, R.LinkCheckPerformed);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.Add(LStatD);
    ID.Add(FStatD);
    ID.AddBoolean(LinkCheckPerformed);
  }
};

struct ASTData {
  const FieldDecl *StModeFD;
  const FieldDecl *StInoFD;
  const FieldDecl *StDevFD;
  QualType StructStatType;
  int64_t O_NOFOLLOWValue;
  bool IsValid;
  void checkValid() {
    IsValid = StModeFD && StInoFD && StDevFD && !StructStatType.isNull();
  }
};

class UnsafeSymlinkTestChecker
    : public Checker<check::PostCall, check::BranchCondition,
                     check::RegionChanges, check::DeadSymbols,
                     check::LiveSymbols> {
  using FnHandler = std::function<void(const UnsafeSymlinkTestChecker *,
                                       const CallEvent &, CheckerContext &)>;

  CallDescriptionMap<FnHandler> Callbacks = {
      {{CDM::CLibrary, {"lstat"}, 2}, &UnsafeSymlinkTestChecker::handleLStat},
      {{CDM::CLibrary, {"open"}, 2}, &UnsafeSymlinkTestChecker::handleOpen},
      {{CDM::CLibrary, {"fstat"}, 2}, &UnsafeSymlinkTestChecker::handleFStat},
  };
  const CallDescriptionSet FileAccessFn{
      {CDM::CLibrary, {"write"}, 3},  {CDM::CLibrary, {"writev"}, 3},
      {CDM::CLibrary, {"pwrite"}, 4}, {CDM::CLibrary, {"read"}, 3},
      {CDM::CLibrary, {"readv"}, 3},  {CDM::CLibrary, {"pread"}, 4},
      {CDM::CLibrary, {"lseek"}, 3},
  };

  const BugType BT{this, "Security error",
                   "Race condition when checking for symbolic link",
                   /*SuppressOnSink=*/false};

  mutable std::optional<ASTData> ASTValues;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkBranchCondition(const Stmt *S, CheckerContext &C) const;
  ProgramStateRef checkRegionChanges(ProgramStateRef State,
                                     const InvalidatedSymbols *Invalidated,
                                     ArrayRef<const MemRegion *> Explicits,
                                     ArrayRef<const MemRegion *> Regions,
                                     const StackFrame *SF,
                                     const CallEvent *Call) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkLiveSymbols(ProgramStateRef State, SymbolReaper &SymReaper) const;

private:
  const SubRegion *castRegionToStructStat(const MemRegion *R,
                                          CheckerContext &C) const;
  const FieldRegion *getStModeRegion(const MemRegion *StatR,
                                     CheckerContext &C) const;
  /// Return the "pretty printed" name of a field of a MemRegion of a
  /// "struct stat" object.
  /// @param PrefixStr Print this before the field name.
  /// @param EmptyStr Use this string if the field can not be pretty-printed.
  std::string getFieldVarString(const MemRegion *StatR,
                                const FieldDecl *StatFieldD,
                                StringRef PrefixStr, StringRef EmptyStr,
                                CheckerContext &C) const;
  StatData getStatData(const SubRegion *StatR, ProgramStateRef State,
                       CheckerContext &C) const;
  const NoteTag *getNoteTag(const MemRegion *R, std::string Message,
                            CheckerContext &C) const;

  void initData(const RecordDecl *StatDecl, const Preprocessor &PP) const;
  void handleLStat(const CallEvent &Call, CheckerContext &C) const;
  void handleOpen(const CallEvent &Call, CheckerContext &C) const;
  void handleFStat(const CallEvent &Call, CheckerContext &C) const;
  void handleFileAccess(const CallEvent &Call, CheckerContext &C) const;
};

} // namespace

/// Data about files where `lstat` was called but not `open`.
REGISTER_MAP_WITH_PROGRAMSTATE(LStatCalledMap, FileNameKey, FileDataLStat)

/// Data about files where `lstat` and `open` was called.
REGISTER_MAP_WITH_PROGRAMSTATE(LStatOpenCalledMap, SymbolRef, FileDataOpened)

const SubRegion *
UnsafeSymlinkTestChecker::castRegionToStructStat(const MemRegion *R,
                                                 CheckerContext &C) const {
  if (!R)
    return nullptr;
  std::optional<const MemRegion *> CastR = C.getStoreManager().castRegion(
      R, C.getASTContext().getPointerType(ASTValues->StructStatType));
  if (CastR) {
    if (const SubRegion *SR = (*CastR)->getAs<SubRegion>())
      return SR;
  }
  return R->getAs<SubRegion>();
}

const FieldRegion *
UnsafeSymlinkTestChecker::getStModeRegion(const MemRegion *StatR,
                                          CheckerContext &C) const {
  return C.getStoreManager().getRegionManager().getFieldRegion(
      ASTValues->StModeFD, castRegionToStructStat(StatR, C));
}

std::string UnsafeSymlinkTestChecker::getFieldVarString(
    const MemRegion *StatR, const FieldDecl *StatFieldD, StringRef PrefixStr,
    StringRef EmptyStr, CheckerContext &C) const {
  const MemRegion *R = C.getStoreManager().getRegionManager().getFieldRegion(
      StatFieldD, castRegionToStructStat(StatR, C));
  if (!R->canPrintPretty())
    return EmptyStr.str();
  SmallString<64> Buf;
  llvm::raw_svector_ostream Out(Buf);
  Out << PrefixStr;
  R->printPretty(Out);
  return Out.str().str();
}

StatData UnsafeSymlinkTestChecker::getStatData(const SubRegion *StatR,
                                               ProgramStateRef State,
                                               CheckerContext &C) const {
  MemRegionManager &RM = C.getStoreManager().getRegionManager();
  auto *StatR1 = castRegionToStructStat(StatR, C);
  auto GetFieldSVal = [&](const FieldDecl *FD) {
    return State->getSVal(RM.getFieldRegion(FD, StatR1));
  };
  return {StatR, GetFieldSVal(ASTValues->StModeFD),
          GetFieldSVal(ASTValues->StInoFD), GetFieldSVal(ASTValues->StDevFD)};
}

const NoteTag *UnsafeSymlinkTestChecker::getNoteTag(const MemRegion *R,
                                                    std::string Message,
                                                    CheckerContext &C) const {
  return C.getNoteTag([this, R, M = std::move(Message)](
                          PathSensitiveBugReport &BR) -> std::string {
    if (&BR.getBugType() == &BT && BR.isInteresting(R))
      return M;
    return "";
  });
}

static const FieldDecl *findField(llvm::StringRef FieldName,
                                  const RecordDecl *RD) {
  auto FoundField =
      llvm::find_if(RD->fields(), [&FieldName](const FieldDecl *F) {
        return F->getNameAsString() == FieldName;
      });
  if (FoundField == RD->fields().end())
    return nullptr;
  return *FoundField;
}

void UnsafeSymlinkTestChecker::initData(const RecordDecl *StatDecl,
                                        const Preprocessor &PP) const {
  if (StatDecl) {
    ASTValues = {findField("st_mode", StatDecl),
                 findField("st_ino", StatDecl),
                 findField("st_dev", StatDecl),
                 StatDecl->getASTContext().getCanonicalTagType(StatDecl),
                 /*O_NOFOLLOWValue=*/0,
                 /*IsValid=*/false};
    if (std::optional<int> Val = tryExpandAsInteger("O_NOFOLLOW", PP))
      ASTValues->O_NOFOLLOWValue = *Val;
  } else {
    ASTValues = {nullptr};
  }
  ASTValues->checkValid();
}

void UnsafeSymlinkTestChecker::handleLStat(const CallEvent &Call,
                                           CheckerContext &C) const {
  if (!ASTValues) {
    const QualType T = Call.parameters()[1]->getType();
    initData(T->isPointerType() ? T->getPointeeType()->getAsRecordDecl()
                                : nullptr,
             C.getPreprocessor());
    if (!ASTValues->IsValid)
      return;
  }

  ProgramStateRef State = C.getState();
  const MemRegion *FNameReg = Call.getArgSVal(0).getAsRegion();
  const auto *StatReg =
      dyn_cast_or_null<SubRegion>(Call.getArgSVal(1).getAsRegion());
  if (!FNameReg || !StatReg || !castRegionToStructStat(StatReg, C))
    return;

  FileNameKey FName(FNameReg);
  State = State->set<LStatCalledMap>(FName,
                                     {getStatData(StatReg, State, C), false});
  C.addTransition(State,
                  getNoteTag(StatReg,
                             (llvm::Twine("File status") +
                              FName.getFileName(" of file ") + " is read here" +
                              getFieldVarString(StatReg, ASTValues->StModeFD,
                                                " into ", "", C) +
                              " before opening the file")
                                 .str(),
                             C));
}

void UnsafeSymlinkTestChecker::handleOpen(const CallEvent &Call,
                                          CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  const MemRegion *FNameReg = Call.getArgSVal(0).getAsRegion();
  FileNameKey FName(FNameReg);
  const FileDataLStat *LStatData = State->get<LStatCalledMap>(FName);
  SymbolRef FileDescSym = Call.getReturnValue().getAsSymbol();
  if (!FNameReg || !LStatData || !FileDescSym)
    return;

  State = State->remove<LStatCalledMap>(FNameReg);

  // If presence of O_NOFOLLOW can be verified, ignore this execution path.
  // Otherwise it can be assumed that O_NOFOLLOW is not set, because when it is
  // set S_ISLNK should not appear ('LinkCheckPerformed' will be false) so that
  // case is still ignored.
  if (ASTValues->O_NOFOLLOWValue != 0) {
    const llvm::APSInt *FlagsValue =
        C.getSValBuilder().getKnownValue(State, Call.getArgSVal(1));
    if (FlagsValue) {
      if (std::optional<int64_t> FVal = FlagsValue->tryExtValue();
          FVal && (*FVal & ASTValues->O_NOFOLLOWValue)) {
        C.addTransition(State);
        return;
      }
    }
  }

  State = State->set<LStatOpenCalledMap>(FileDescSym,
                                         {LStatData->LStatD,
                                          {nullptr, SVal{}, SVal{}, SVal{}},
                                          LStatData->LinkCheckPerformed,
                                          FName});

  C.addTransition(State, getNoteTag(LStatData->LStatD.Region,
                                    (llvm::Twine("File") +
                                     FName.getFileName(" ") + " is opened here")
                                        .str(),
                                    C));
}

void UnsafeSymlinkTestChecker::handleFStat(const CallEvent &Call,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef FileDescSym = Call.getArgSVal(0).getAsSymbol();
  const auto *FStatReg =
      dyn_cast_or_null<SubRegion>(Call.getArgSVal(1).getAsRegion());
  if (!FileDescSym || !FStatReg || !castRegionToStructStat(FStatReg, C))
    return;

  const FileDataOpened *FileData = State->get<LStatOpenCalledMap>(FileDescSym);
  if (!FileData)
    return;

  State = State->set<LStatOpenCalledMap>(
      FileDescSym, {FileData->LStatD, getStatData(FStatReg, State, C),
                    FileData->LinkCheckPerformed, FileData->FName});
  C.addTransition(
      State,
      getNoteTag(
          FStatReg,
          (llvm::Twine("File status") +
           FileData->FName.getFileName(" of file ") + " is read here" +
           getFieldVarString(FStatReg, ASTValues->StModeFD, " into ", "", C) +
           " after opening the file")
              .str(),
          C));
}

void UnsafeSymlinkTestChecker::handleFileAccess(const CallEvent &Call,
                                                CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  SymbolRef FileDescSym = Call.getArgSVal(0).getAsSymbol();
  if (!FileDescSym)
    return;

  const FileDataOpened *FileData = State->get<LStatOpenCalledMap>(FileDescSym);
  if (!FileData)
    return;

  State = State->remove<LStatOpenCalledMap>(FileDescSym);
  if (!FileData->LinkCheckPerformed) {
    C.addTransition(State);
    return;
  }

  auto CheckEqual = [State, &C](SVal V1, SVal V2) {
    auto DefVal1 = V1.getAs<DefinedOrUnknownSVal>();
    auto DefVal2 = V2.getAs<DefinedOrUnknownSVal>();
    if (!DefVal1 || !DefVal2)
      return false;
    DefinedOrUnknownSVal EQV =
        C.getSValBuilder().evalEQ(State, *DefVal1, *DefVal2);
    auto [EQTrue, EQFalse] = State->assume(EQV);
    return EQTrue && !EQFalse;
  };
  if (FileData->FStatD.Region &&
      CheckEqual(FileData->FStatD.StModeVal, FileData->LStatD.StModeVal) &&
      CheckEqual(FileData->FStatD.StInoVal, FileData->LStatD.StInoVal) &&
      CheckEqual(FileData->FStatD.StDevVal, FileData->LStatD.StDevVal)) {
    C.addTransition(State);
    return;
  }

  if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
    auto R = std::make_unique<PathSensitiveBugReport>(
        BT,
        (llvm::Twine("File") + FileData->FName.getFileName(" ") +
         " might have been changed between call to 'lstat' and 'open' "
         "therefore " +
         getFieldVarString(FileData->LStatD.Region, ASTValues->StModeFD, "",
                           "the file status value", C) +
         " may not contain the state of the file at open")
            .str(),
        N);
    R->addRange(Call.getSourceRange());
    R->markInteresting(FileData->LStatD.Region);
    R->markInteresting(FileData->FStatD.Region);
    C.emitReport(std::move(R));
    return;
  }
}

void UnsafeSymlinkTestChecker::checkPostCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  if (ASTValues && !ASTValues->IsValid)
    return;

  if (const FnHandler *Fn = Callbacks.lookup(Call))
    (*Fn)(this, Call, C);
  else if (FileAccessFn.contains(Call))
    handleFileAccess(Call, C);
}

namespace {
class FindMacroVisitor : public ConstStmtVisitor<FindMacroVisitor, bool> {
  const CheckerContext &C;
  ProgramStateRef State;
  const MemRegion *LStatInfoStModeReg;

  bool VisitChildren(const Stmt *S) {
    for (const Stmt *Child : S->children())
      if (Child && Visit(Child))
        return true;
    return false;
  }

public:
  FindMacroVisitor(const CheckerContext &C, ProgramStateRef State,
                   const MemRegion *LStatInfoStModeReg)
      : C(C), State(State), LStatInfoStModeReg(LStatInfoStModeReg) {}
  bool VisitStmt(const Stmt *S) { return VisitChildren(S); }
  bool VisitExpr(const Expr *E) {
    if (check(E))
      return true;
    return VisitChildren(E);
  }

private:
  bool check(const Expr *E) {
    const MemRegion *R = State->getSVal(E, C.getStackFrame()).getAsRegion();
    if (R != LStatInfoStModeReg)
      return false;
    SourceLocation BL = E->getBeginLoc();
    if (!BL.isMacroID())
      return false;
    const SourceManager &SM = C.getASTContext().getSourceManager();
    SourceLocation StartL;
    if (!SM.isMacroArgExpansion(BL, &StartL))
      return false;
    StringRef MacroName = Lexer::getImmediateMacroName(BL, SM, C.getLangOpts());
    return MacroName == "S_ISLNK";
  }
};
} // end anonymous namespace

void UnsafeSymlinkTestChecker::checkBranchCondition(const Stmt *S,
                                                    CheckerContext &C) const {
  if (!ASTValues || !ASTValues->IsValid)
    return;

  ExplodedNode *NewNode = C.getPredecessor();
  LStatCalledMapTy LStatCalled = NewNode->getState()->get<LStatCalledMap>();
  for (auto I : LStatCalled) {
    const FieldRegion *FR = getStModeRegion(I.second.LStatD.Region, C);
    ProgramStateRef State = NewNode->getState();
    FindMacroVisitor FindS_ISLNK(C, State, FR);
    if (FindS_ISLNK.Visit(S)) {
      State = State->set<LStatCalledMap>(I.first, {I.second.LStatD, true});
      NewNode = C.addTransition(
          State, NewNode,
          getNoteTag(I.second.LStatD.Region,
                     (llvm::Twine(getFieldVarString(I.second.LStatD.Region,
                                                    ASTValues->StModeFD, "",
                                                    "File status value", C)) +
                      " is checked here for symbolic link")
                         .str(),
                     C));
    }
  }

  LStatOpenCalledMapTy LStatOpenCalled =
      NewNode->getState()->get<LStatOpenCalledMap>();
  for (auto I : LStatOpenCalled) {
    if (I.second.LinkCheckPerformed)
      continue;

    const FieldRegion *FR = getStModeRegion(I.second.LStatD.Region, C);
    ProgramStateRef State = NewNode->getState();
    FindMacroVisitor FindS_ISLNK(C, State, FR);
    if (FindS_ISLNK.Visit(S)) {
      State = State->set<LStatOpenCalledMap>(
          I.first, {I.second.LStatD, I.second.FStatD, true, I.second.FName});
      NewNode = C.addTransition(
          State, NewNode,
          getNoteTag(I.second.LStatD.Region,
                     (llvm::Twine(getFieldVarString(I.second.LStatD.Region,
                                                    ASTValues->StModeFD, "",
                                                    "File status value", C)) +
                      " is checked here for symbolic link")
                         .str(),
                     C));
    }
  }
}

ProgramStateRef UnsafeSymlinkTestChecker::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> Explicits, ArrayRef<const MemRegion *> Regions,
    const StackFrame *SF, const CallEvent *Call) const {
  if (Call && Callbacks.lookup(*Call))
    return State;

  if (Invalidated) {
    for (SymbolRef I : *Invalidated)
      State = State->remove<LStatOpenCalledMap>(I);
  }
  llvm::SmallPtrSet<const MemRegion *, 4> InvalidatedR;
  for (const MemRegion *R : Regions)
    InvalidatedR.insert(R);
  for (auto I : State->get<LStatCalledMap>())
    if (InvalidatedR.contains(I.second.LStatD.Region))
      State = State->remove<LStatCalledMap>(I.first);
  for (auto I : State->get<LStatOpenCalledMap>())
    if (InvalidatedR.contains(I.second.LStatD.Region) ||
        InvalidatedR.contains(I.second.FStatD.Region))
      State = State->remove<LStatOpenCalledMap>(I.first);
  return State;
}

void UnsafeSymlinkTestChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                                CheckerContext &C) const {
  if (!ASTValues || !ASTValues->IsValid)
    return;

  ProgramStateRef State = C.getState();
  for (auto I : State->get<LStatCalledMap>()) {
    if (const auto *SymReg =
            dyn_cast_or_null<SymbolicRegion>(I.first.getRegionOrNull());
        SymReg && SymReg->getSymbol() && SymReaper.isDead(SymReg->getSymbol()))
      State = State->remove<LStatCalledMap>(I.first.getRegionOrNull());
  }
  for (auto I : State->get<LStatOpenCalledMap>()) {
    if (SymReaper.isDead(I.first))
      State = State->remove<LStatOpenCalledMap>(I.first);
  }

  C.addTransition(State);
}

void UnsafeSymlinkTestChecker::checkLiveSymbols(ProgramStateRef State,
                                                SymbolReaper &SymReaper) const {
  if (!ASTValues || !ASTValues->IsValid)
    return;

  // The SVal objects at these regions may be needed at checkFileAccess later.
  // These values may expire before that point if the parent region is not
  // marked as live.
  for (auto I : State->get<LStatOpenCalledMap>()) {
    if (I.second.FStatD.Region)
      SymReaper.markLive(I.second.FStatD.Region);
    if (I.second.LStatD.Region)
      SymReaper.markLive(I.second.LStatD.Region);
  }
}

void ento::registerUnsafeSymlinkTestChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnsafeSymlinkTestChecker>();
}

bool ento::shouldRegisterUnsafeSymlinkTestChecker(const CheckerManager &mgr) {
  return true;
}
