//===-- UnsafeSymlinkTestChecker.cpp ------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker that checks for unsafe symlink detection. This checks for
// 2 related conditions:
// - File status is read and used to detect symlink before the file is opened.
//   The file can be changed asynchronously between reading the status data and
//   opening the file, so this check is not safe to use.
// - To fix the previous issue, the file status can be read after the open too
//   and compared to the previous value. If it did not change, the symlink
//   status is safely determined (the file can not be changed externally after
//   it was opened). The checker can detect a missing comparison of the "before"
//   and "after" status values.
// (In all cases use of the O_NOFOLLOW flag at 'open' prevents the warning.)
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
struct FileNameKey {
  std::string FileNameStr;
  const MemRegion *Region = nullptr;

  FileNameKey(const MemRegion *R) {
    R = R->StripCasts();
    if (const auto *SR = dyn_cast<StringRegion>(R))
      FileNameStr = SR->getStringLiteral()->getString();
    else
      Region = R;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddString(FileNameStr);
    ID.AddPointer(Region);
  }

  bool operator==(const FileNameKey &RHS) const {
    return FileNameStr == RHS.FileNameStr && Region == RHS.Region;
  }

  bool operator<(const FileNameKey &RHS) const {
    if (!Region && !RHS.Region)
      return FileNameStr < RHS.FileNameStr;
    return Region < RHS.Region;
  }

  std::string getFileName(llvm::StringRef PrefixStr) const {
    if (!Region)
      return (llvm::Twine(PrefixStr) + "'" + FileNameStr + "'").str();
    return "";
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
    return Region == D.Region && StModeVal == D.StModeVal &&
           StInoVal == D.StInoVal && StDevVal == D.StDevVal;
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Region);
    StModeVal.Profile(ID);
    StInoVal.Profile(ID);
    StDevVal.Profile(ID);
  }
};

/// Data about a file after `lstat` (but not `open`) was called.
struct FileDataLStat {
  /// Information about the `stat` structure that was passed to `lstat`.
  StatData LStatD;
  /// Indicates if a test for symbolic link on the `st_mode` field of the `stat`
  /// structure was performed, using the `S_ISLNK` macro.
  bool LinkCheckPerformed;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    LStatD.Profile(ID);
    ID.AddBoolean(LinkCheckPerformed);
  }

  bool operator==(const FileDataLStat &R) const {
    return LStatD == R.LStatD && LinkCheckPerformed == R.LinkCheckPerformed;
  }
};

/// Data about a file after `lstat` and `open` was called (no symbolic link test
/// with `S_ISLNK` was performed in between).
struct FileDataOpened {
  /// Information about the `stat` structure that was passed to `lstat`.
  StatData LStatD;
  /// Information about the `stat` structure that was passed to `fstat`.
  StatData FStatD;
  /// Data about the file name (this is used for checker messages).
  FileNameKey FName;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    LStatD.Profile(ID);
    FStatD.Profile(ID);
  }

  bool operator==(const FileDataOpened &R) const {
    return LStatD == R.LStatD && FStatD == R.FStatD;
  }
};

struct StatFieldsDecl {
  const FieldDecl *StModeFD;
  const FieldDecl *StInoFD;
  const FieldDecl *StDevFD;

  bool isValid() const { return StModeFD && StInoFD && StDevFD; }
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
                     check::RegionChanges, check::DeadSymbols> {
  const CallDescription LStatFn{CDM::CLibrary, {"lstat"}, 2};
  const CallDescription OpenFn{CDM::CLibrary, {"open"}, 2};
  const CallDescription FStatFn{CDM::CLibrary, {"fstat"}, 2};
  const CallDescriptionSet FileAccessFn{
      {CDM::CLibrary, {"write"}, 3},  {CDM::CLibrary, {"writev"}, 3},
      {CDM::CLibrary, {"pwrite"}, 4}, {CDM::CLibrary, {"read"}, 3},
      {CDM::CLibrary, {"readv"}, 3},  {CDM::CLibrary, {"pread"}, 4},
      {CDM::CLibrary, {"lseek"}, 3}};

  const BugType BT{this, "Security error", "Incorrect check for symbolic link",
                   false};

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

private:
  const SubRegion *castRegionToStructStat(const MemRegion *R,
                                          CheckerContext &C) const {
    if (!R)
      return nullptr;
    std::optional<const MemRegion *> CastR = C.getStoreManager().castRegion(
        R, C.getASTContext().getPointerType(ASTValues->StructStatType));
    if (!CastR)
      return R->getAs<SubRegion>();
    const SubRegion *SR = (*CastR)->getAs<SubRegion>();
    return SR ? SR : R->getAs<SubRegion>();
  }
  StatData getStatData(const SubRegion *StatR, ProgramStateRef State,
                       CheckerContext &C) const {
    MemRegionManager &RM = C.getStoreManager().getRegionManager();
    auto *StatR1 = castRegionToStructStat(StatR, C);
    auto GetFieldSVal = [&](const FieldDecl *FD) {
      return State->getSVal(RM.getFieldRegion(FD, StatR1));
    };
    return {StatR, GetFieldSVal(ASTValues->StModeFD),
            GetFieldSVal(ASTValues->StInoFD), GetFieldSVal(ASTValues->StDevFD)};
  }
  const NoteTag *getNoteTag(const MemRegion *R, std::string Message,
                            CheckerContext &C) const;
  void initData(const RecordDecl *StatDecl, const Preprocessor &PP) const;
};

} // end anonymous namespace

/// Data about files where `lstat` was called but not `open`.
REGISTER_MAP_WITH_PROGRAMSTATE(LStatCalledMap, FileNameKey, FileDataLStat)

/// Data about files where `lstat` and `open` was called.
REGISTER_MAP_WITH_PROGRAMSTATE(LStatOpenCalledMap, SymbolRef, FileDataOpened)

const NoteTag *UnsafeSymlinkTestChecker::getNoteTag(const MemRegion *R,
                                                    std::string Message,
                                                    CheckerContext &C) const {
  return C.getNoteTag(
      [this, R, Message](PathSensitiveBugReport &BR) -> std::string {
        if (BR.isInteresting(R) && &BR.getBugType() == &BT)
          return Message;
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
                 0,
                 false};
    if (std::optional<int> Val = tryExpandAsInteger("O_NOFOLLOW", PP))
      ASTValues->O_NOFOLLOWValue = *Val;
  } else {
    ASTValues = {nullptr};
  }
  ASTValues->checkValid();
}

void UnsafeSymlinkTestChecker::checkPostCall(const CallEvent &Call,
                                             CheckerContext &C) const {
  if (ASTValues && !ASTValues->IsValid)
    return;

  ProgramStateRef State = C.getState();

  if (LStatFn.matches(Call)) {
    if (!ASTValues) {
      initData(
          Call.parameters()[1]->getType()->getPointeeType()->getAsRecordDecl(),
          C.getPreprocessor());
      if (!ASTValues->IsValid)
        return;
    }

    const MemRegion *FNameReg = Call.getArgSVal(0).getAsRegion();
    const auto *StatReg =
        dyn_cast_or_null<SubRegion>(Call.getArgSVal(1).getAsRegion());
    if (!FNameReg || !StatReg)
      return;

    FileNameKey FName(FNameReg);
    State = State->set<LStatCalledMap>(FName,
                                       {getStatData(StatReg, State, C), false});
    C.addTransition(State, getNoteTag(StatReg,
                                      (llvm::Twine("File status") +
                                       FName.getFileName(" of file ") +
                                       " is read here before opening the file")
                                          .str(),
                                      C));
    return;
  }

  if (OpenFn.matches(Call)) {
    const MemRegion *FNameReg = Call.getArgSVal(0).getAsRegion();
    FileNameKey FName(FNameReg);
    const FileDataLStat *LStatData = State->get<LStatCalledMap>(FName);
    SymbolRef FileDescSym = Call.getReturnValue().getAsSymbol();
    if (!FNameReg || !LStatData || !FileDescSym)
      return;

    State = State->remove<LStatCalledMap>(FNameReg);

    if (ASTValues->O_NOFOLLOWValue != 0) {
      const llvm::APSInt *FlagsValue =
          C.getSValBuilder().getKnownValue(State, Call.getArgSVal(1));
      if (!FlagsValue) {
        C.addTransition(State);
        return;
      }
      if (std::optional<int64_t> FVal = FlagsValue->tryExtValue();
          FVal && (*FVal & ASTValues->O_NOFOLLOWValue)) {
        C.addTransition(State);
        return;
      }
    }

    if (!LStatData->LinkCheckPerformed) {
      State = State->set<LStatOpenCalledMap>(
          FileDescSym,
          {LStatData->LStatD, {nullptr, SVal{}, SVal{}, SVal{}}, FName});
    } else {
      if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
        auto R = std::make_unique<PathSensitiveBugReport>(
            BT,
            (llvm::Twine("Inaccurate check for symbolic link status of file") +
             FName.getFileName(" "))
                .str(),
            N);
        R->addNote("The file can be manipulated externally between calling "
                   "'lstat' and opening the file",
                   {Call.getSourceRange().getBegin(), C.getSourceManager()});
        R->addRange(Call.getSourceRange());
        R->markInteresting(LStatData->LStatD.Region);
        C.emitReport(std::move(R));
        return;
      }
    }
  }

  if (FStatFn.matches(Call)) {
    SymbolRef FileDescSym = Call.getArgSVal(0).getAsSymbol();
    const auto *FStatReg =
        dyn_cast_or_null<SubRegion>(Call.getArgSVal(1).getAsRegion());
    if (!FileDescSym || !FStatReg)
      return;
    const FileDataOpened *FileData =
        State->get<LStatOpenCalledMap>(FileDescSym);
    if (!FileData)
      return;
    State = State->set<LStatOpenCalledMap>(
        FileDescSym,
        {FileData->LStatD, getStatData(FStatReg, State, C), FileData->FName});
    C.addTransition(State,
                    getNoteTag(FStatReg,
                               (llvm::Twine("File status") +
                                FileData->FName.getFileName(" of file ") +
                                " is read here after opening the file")
                                   .str(),
                               C));
    return;
  }

  if (FileAccessFn.contains(Call)) {
    SymbolRef FileDescSym = Call.getArgSVal(0).getAsSymbol();
    if (!FileDescSym)
      return;
    const FileDataOpened *FileData =
        State->get<LStatOpenCalledMap>(FileDescSym);
    if (!FileData)
      return;
    State = State->remove<LStatOpenCalledMap>(FileDescSym);
    if (ExplodedNode *N = C.generateNonFatalErrorNode(State)) {
      auto R = std::make_unique<PathSensitiveBugReport>(
          BT,
          (llvm::Twine("Possibly missing check for external change of file") +
           FileData->FName.getFileName(" "))
              .str(),
          N);
      R->addNote(
          "File status was obtained before and after opening the file which "
          "indicates possible intent of a safe check for symbolic link",
          {Call.getSourceRange().getBegin(), C.getSourceManager()});
      R->addNote("For a safe check the fields 'st_mode', 'st_ino' and 'st_dev' "
                 "before and after open should be checked for equality",
                 {Call.getSourceRange().getBegin(), C.getSourceManager()});
      R->addRange(Call.getSourceRange());
      R->markInteresting(FileData->LStatD.Region);
      R->markInteresting(FileData->FStatD.Region);
      C.emitReport(std::move(R));
      return;
    }
  }

  C.addTransition(State);
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
    const FieldRegion *FR =
        C.getStoreManager().getRegionManager().getFieldRegion(
            ASTValues->StModeFD,
            castRegionToStructStat(I.second.LStatD.Region, C));
    ProgramStateRef State = NewNode->getState();
    FindMacroVisitor FindS_ISLNK(C, State, FR);
    if (FindS_ISLNK.Visit(S)) {
      State = State->set<LStatCalledMap>(I.first, {I.second.LStatD, true});
      NewNode =
          C.addTransition(State, NewNode,
                          getNoteTag(I.second.LStatD.Region,
                                     (llvm::Twine("Possible test if file") +
                                      I.first.getFileName(" ") +
                                      " is a symbolic link detected here")
                                         .str(),
                                     C));
    }
  }

  ProgramStateRef State = NewNode->getState();
  LStatOpenCalledMapTy LStatOpenCalled = State->get<LStatOpenCalledMap>();
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
  for (auto I : LStatOpenCalled) {
    if (I.second.FStatD.Region)
      if (CheckEqual(I.second.FStatD.StModeVal, I.second.LStatD.StModeVal) &&
          CheckEqual(I.second.FStatD.StInoVal, I.second.LStatD.StInoVal) &&
          CheckEqual(I.second.FStatD.StDevVal, I.second.LStatD.StDevVal))
        State = State->remove<LStatOpenCalledMap>(I.first);
  }

  C.addTransition(State, NewNode);
}

ProgramStateRef UnsafeSymlinkTestChecker::checkRegionChanges(
    ProgramStateRef State, const InvalidatedSymbols *Invalidated,
    ArrayRef<const MemRegion *> Explicits, ArrayRef<const MemRegion *> Regions,
    const StackFrame *SF, const CallEvent *Call) const {
  if (Call && (LStatFn.matches(*Call) || OpenFn.matches(*Call) ||
               FStatFn.matches(*Call)))
    return State;

  if (Invalidated) {
    for (SymbolRef I : *Invalidated)
      State = State->remove<LStatOpenCalledMap>(I);
  }
  llvm::SmallPtrSet<const MemRegion *, 4> InvalidatedR;
  for (const MemRegion *R : Regions)
    InvalidatedR.insert(R);
  for (auto I : State->get<LStatCalledMap>())
    if (!I.second.LinkCheckPerformed &&
        InvalidatedR.contains(I.second.LStatD.Region))
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
    if (const auto *SymReg = dyn_cast_or_null<SymbolicRegion>(I.first.Region);
        SymReg && SymReg->getSymbol() && SymReaper.isDead(SymReg->getSymbol()))
      State = State->remove<LStatCalledMap>(I.first.Region);
  }
  for (auto I : State->get<LStatOpenCalledMap>()) {
    if (SymReaper.isDead(I.first))
      State = State->remove<LStatOpenCalledMap>(I.first);
  }

  C.addTransition(State);
}

void ento::registerUnsafeSymlinkTestChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnsafeSymlinkTestChecker>();
}

bool ento::shouldRegisterUnsafeSymlinkTestChecker(const CheckerManager &mgr) {
  return true;
}
