//===--- Diagnostic.cpp - C Language Family Diagnostic Handling -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Diagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include <cctype>

using namespace clang;

static void DummyArgToStringFn(DiagnosticsEngine::ArgumentKind AK, intptr_t QT,
                               const char *Modifier, unsigned ML,
                               const char *Argument, unsigned ArgLen,
                               const DiagnosticsEngine::ArgumentValue *PrevArgs,
                               unsigned NumPrevArgs,
                               SmallVectorImpl<char> &Output,
                               void *Cookie,
                               ArrayRef<intptr_t> QualTypeVals) {
  const char *Str = "<can't format argument>";
  Output.append(Str, Str+strlen(Str));
}


DiagnosticsEngine::DiagnosticsEngine(
                       const IntrusiveRefCntPtr<DiagnosticIDs> &diags,
                       DiagnosticOptions *DiagOpts,       
                       DiagnosticConsumer *client, bool ShouldOwnClient)
  : Diags(diags), DiagOpts(DiagOpts), Client(client),
    OwnsDiagClient(ShouldOwnClient), SourceMgr(0) {
  ArgToStringFn = DummyArgToStringFn;
  ArgToStringCookie = 0;

  AllExtensionsSilenced = 0;
  IgnoreAllWarnings = false;
  WarningsAsErrors = false;
  EnableAllWarnings = false;
  ErrorsAsFatal = false;
  SuppressSystemWarnings = false;
  SuppressAllDiagnostics = false;
  ElideType = true;
  PrintTemplateTree = false;
  ShowColors = false;
  ShowOverloads = Ovl_All;
  ExtBehavior = Ext_Ignore;

  ErrorLimit = 0;
  TemplateBacktraceLimit = 0;
  ConstexprBacktraceLimit = 0;

  Reset();
}

DiagnosticsEngine::~DiagnosticsEngine() {
  if (OwnsDiagClient)
    delete Client;
}

void DiagnosticsEngine::setClient(DiagnosticConsumer *client,
                                  bool ShouldOwnClient) {
  if (OwnsDiagClient && Client)
    delete Client;
  
  Client = client;
  OwnsDiagClient = ShouldOwnClient;
}

void DiagnosticsEngine::pushMappings(SourceLocation Loc) {
  DiagStateOnPushStack.push_back(GetCurDiagState());
}

bool DiagnosticsEngine::popMappings(SourceLocation Loc) {
  if (DiagStateOnPushStack.empty())
    return false;

  if (DiagStateOnPushStack.back() != GetCurDiagState()) {
    // State changed at some point between push/pop.
    PushDiagStatePoint(DiagStateOnPushStack.back(), Loc);
  }
  DiagStateOnPushStack.pop_back();
  return true;
}

void DiagnosticsEngine::Reset() {
  ErrorOccurred = false;
  FatalErrorOccurred = false;
  UnrecoverableErrorOccurred = false;
  
  NumWarnings = 0;
  NumErrors = 0;
  NumErrorsSuppressed = 0;
  TrapNumErrorsOccurred = 0;
  TrapNumUnrecoverableErrorsOccurred = 0;
  
  CurDiagID = ~0U;
  // Set LastDiagLevel to an "unset" state. If we set it to 'Ignored', notes
  // using a DiagnosticsEngine associated to a translation unit that follow
  // diagnostics from a DiagnosticsEngine associated to anoter t.u. will not be
  // displayed.
  LastDiagLevel = (DiagnosticIDs::Level)-1;
  DelayedDiagID = 0;

  // Clear state related to #pragma diagnostic.
  DiagStates.clear();
  DiagStatePoints.clear();
  DiagStateOnPushStack.clear();

  // Create a DiagState and DiagStatePoint representing diagnostic changes
  // through command-line.
  DiagStates.push_back(DiagState());
  DiagStatePoints.push_back(DiagStatePoint(&DiagStates.back(), FullSourceLoc()));
}

void DiagnosticsEngine::SetDelayedDiagnostic(unsigned DiagID, StringRef Arg1,
                                             StringRef Arg2) {
  if (DelayedDiagID)
    return;

  DelayedDiagID = DiagID;
  DelayedDiagArg1 = Arg1.str();
  DelayedDiagArg2 = Arg2.str();
}

void DiagnosticsEngine::ReportDelayed() {
  Report(DelayedDiagID) << DelayedDiagArg1 << DelayedDiagArg2;
  DelayedDiagID = 0;
  DelayedDiagArg1.clear();
  DelayedDiagArg2.clear();
}

DiagnosticsEngine::DiagStatePointsTy::iterator
DiagnosticsEngine::GetDiagStatePointForLoc(SourceLocation L) const {
  assert(!DiagStatePoints.empty());
  assert(DiagStatePoints.front().Loc.isInvalid() &&
         "Should have created a DiagStatePoint for command-line");

  if (!SourceMgr)
    return DiagStatePoints.end() - 1;

  FullSourceLoc Loc(L, *SourceMgr);
  if (Loc.isInvalid())
    return DiagStatePoints.end() - 1;

  DiagStatePointsTy::iterator Pos = DiagStatePoints.end();
  FullSourceLoc LastStateChangePos = DiagStatePoints.back().Loc;
  if (LastStateChangePos.isValid() &&
      Loc.isBeforeInTranslationUnitThan(LastStateChangePos))
    Pos = std::upper_bound(DiagStatePoints.begin(), DiagStatePoints.end(),
                           DiagStatePoint(0, Loc));
  --Pos;
  return Pos;
}

void DiagnosticsEngine::setDiagnosticMapping(diag::kind Diag, diag::Mapping Map,
                                             SourceLocation L) {
  assert(Diag < diag::DIAG_UPPER_LIMIT &&
         "Can only map builtin diagnostics");
  assert((Diags->isBuiltinWarningOrExtension(Diag) ||
          (Map == diag::MAP_FATAL || Map == diag::MAP_ERROR)) &&
         "Cannot map errors into warnings!");
  assert(!DiagStatePoints.empty());
  assert((L.isInvalid() || SourceMgr) && "No SourceMgr for valid location");

  FullSourceLoc Loc = SourceMgr? FullSourceLoc(L, *SourceMgr) : FullSourceLoc();
  FullSourceLoc LastStateChangePos = DiagStatePoints.back().Loc;
  // Don't allow a mapping to a warning override an error/fatal mapping.
  if (Map == diag::MAP_WARNING) {
    DiagnosticMappingInfo &Info = GetCurDiagState()->getOrAddMappingInfo(Diag);
    if (Info.getMapping() == diag::MAP_ERROR ||
        Info.getMapping() == diag::MAP_FATAL)
      Map = Info.getMapping();
  }
  DiagnosticMappingInfo MappingInfo = makeMappingInfo(Map, L);

  // Common case; setting all the diagnostics of a group in one place.
  if (Loc.isInvalid() || Loc == LastStateChangePos) {
    GetCurDiagState()->setMappingInfo(Diag, MappingInfo);
    return;
  }

  // Another common case; modifying diagnostic state in a source location
  // after the previous one.
  if ((Loc.isValid() && LastStateChangePos.isInvalid()) ||
      LastStateChangePos.isBeforeInTranslationUnitThan(Loc)) {
    // A diagnostic pragma occurred, create a new DiagState initialized with
    // the current one and a new DiagStatePoint to record at which location
    // the new state became active.
    DiagStates.push_back(*GetCurDiagState());
    PushDiagStatePoint(&DiagStates.back(), Loc);
    GetCurDiagState()->setMappingInfo(Diag, MappingInfo);
    return;
  }

  // We allow setting the diagnostic state in random source order for
  // completeness but it should not be actually happening in normal practice.

  DiagStatePointsTy::iterator Pos = GetDiagStatePointForLoc(Loc);
  assert(Pos != DiagStatePoints.end());

  // Update all diagnostic states that are active after the given location.
  for (DiagStatePointsTy::iterator
         I = Pos+1, E = DiagStatePoints.end(); I != E; ++I) {
    GetCurDiagState()->setMappingInfo(Diag, MappingInfo);
  }

  // If the location corresponds to an existing point, just update its state.
  if (Pos->Loc == Loc) {
    GetCurDiagState()->setMappingInfo(Diag, MappingInfo);
    return;
  }

  // Create a new state/point and fit it into the vector of DiagStatePoints
  // so that the vector is always ordered according to location.
  Pos->Loc.isBeforeInTranslationUnitThan(Loc);
  DiagStates.push_back(*Pos->State);
  DiagState *NewState = &DiagStates.back();
  GetCurDiagState()->setMappingInfo(Diag, MappingInfo);
  DiagStatePoints.insert(Pos+1, DiagStatePoint(NewState,
                                               FullSourceLoc(Loc, *SourceMgr)));
}

bool DiagnosticsEngine::setDiagnosticGroupMapping(
  StringRef Group, diag::Mapping Map, SourceLocation Loc)
{
  // Get the diagnostics in this group.
  llvm::SmallVector<diag::kind, 8> GroupDiags;
  if (Diags->getDiagnosticsInGroup(Group, GroupDiags))
    return true;

  // Set the mapping.
  for (unsigned i = 0, e = GroupDiags.size(); i != e; ++i)
    setDiagnosticMapping(GroupDiags[i], Map, Loc);

  return false;
}

void DiagnosticsEngine::setDiagnosticWarningAsError(diag::kind Diag,
                                                    bool Enabled) {
  // If we are enabling this feature, just set the diagnostic mappings to map to
  // errors.
  if (Enabled) 
    setDiagnosticMapping(Diag, diag::MAP_ERROR, SourceLocation());

  // Otherwise, we want to set the diagnostic mapping's "no Werror" bit, and
  // potentially downgrade anything already mapped to be a warning.
  DiagnosticMappingInfo &Info = GetCurDiagState()->getOrAddMappingInfo(Diag);

  if (Info.getMapping() == diag::MAP_ERROR ||
      Info.getMapping() == diag::MAP_FATAL)
    Info.setMapping(diag::MAP_WARNING);

  Info.setNoWarningAsError(true);
}

bool DiagnosticsEngine::setDiagnosticGroupWarningAsError(StringRef Group,
                                                         bool Enabled) {
  // If we are enabling this feature, just set the diagnostic mappings to map to
  // errors.
  if (Enabled)
    return setDiagnosticGroupMapping(Group, diag::MAP_ERROR);

  // Otherwise, we want to set the diagnostic mapping's "no Werror" bit, and
  // potentially downgrade anything already mapped to be a warning.

  // Get the diagnostics in this group.
  llvm::SmallVector<diag::kind, 8> GroupDiags;
  if (Diags->getDiagnosticsInGroup(Group, GroupDiags))
    return true;

  // Perform the mapping change.
  for (unsigned i = 0, e = GroupDiags.size(); i != e; ++i) {
    DiagnosticMappingInfo &Info = GetCurDiagState()->getOrAddMappingInfo(
      GroupDiags[i]);

    if (Info.getMapping() == diag::MAP_ERROR ||
        Info.getMapping() == diag::MAP_FATAL)
      Info.setMapping(diag::MAP_WARNING);

    Info.setNoWarningAsError(true);
  }

  return false;
}

void DiagnosticsEngine::setDiagnosticErrorAsFatal(diag::kind Diag,
                                                  bool Enabled) {
  // If we are enabling this feature, just set the diagnostic mappings to map to
  // errors.
  if (Enabled)
    setDiagnosticMapping(Diag, diag::MAP_FATAL, SourceLocation());
  
  // Otherwise, we want to set the diagnostic mapping's "no Werror" bit, and
  // potentially downgrade anything already mapped to be a warning.
  DiagnosticMappingInfo &Info = GetCurDiagState()->getOrAddMappingInfo(Diag);
  
  if (Info.getMapping() == diag::MAP_FATAL)
    Info.setMapping(diag::MAP_ERROR);
  
  Info.setNoErrorAsFatal(true);
}

bool DiagnosticsEngine::setDiagnosticGroupErrorAsFatal(StringRef Group,
                                                       bool Enabled) {
  // If we are enabling this feature, just set the diagnostic mappings to map to
  // fatal errors.
  if (Enabled)
    return setDiagnosticGroupMapping(Group, diag::MAP_FATAL);

  // Otherwise, we want to set the diagnostic mapping's "no Werror" bit, and
  // potentially downgrade anything already mapped to be an error.

  // Get the diagnostics in this group.
  llvm::SmallVector<diag::kind, 8> GroupDiags;
  if (Diags->getDiagnosticsInGroup(Group, GroupDiags))
    return true;

  // Perform the mapping change.
  for (unsigned i = 0, e = GroupDiags.size(); i != e; ++i) {
    DiagnosticMappingInfo &Info = GetCurDiagState()->getOrAddMappingInfo(
      GroupDiags[i]);

    if (Info.getMapping() == diag::MAP_FATAL)
      Info.setMapping(diag::MAP_ERROR);

    Info.setNoErrorAsFatal(true);
  }

  return false;
}

void DiagnosticsEngine::setMappingToAllDiagnostics(diag::Mapping Map,
                                                   SourceLocation Loc) {
  // Get all the diagnostics.
  llvm::SmallVector<diag::kind, 64> AllDiags;
  Diags->getAllDiagnostics(AllDiags);

  // Set the mapping.
  for (unsigned i = 0, e = AllDiags.size(); i != e; ++i)
    if (Diags->isBuiltinWarningOrExtension(AllDiags[i]))
      setDiagnosticMapping(AllDiags[i], Map, Loc);
}

void DiagnosticsEngine::Report(const StoredDiagnostic &storedDiag) {
  assert(CurDiagID == ~0U && "Multiple diagnostics in flight at once!");

  CurDiagLoc = storedDiag.getLocation();
  CurDiagID = storedDiag.getID();
  NumDiagArgs = 0;

  NumDiagRanges = storedDiag.range_size();
  assert(NumDiagRanges < DiagnosticsEngine::MaxRanges &&
         "Too many arguments to diagnostic!");
  unsigned i = 0;
  for (StoredDiagnostic::range_iterator
         RI = storedDiag.range_begin(),
         RE = storedDiag.range_end(); RI != RE; ++RI)
    DiagRanges[i++] = *RI;

  assert(NumDiagRanges < DiagnosticsEngine::MaxFixItHints &&
         "Too many arguments to diagnostic!");
  NumDiagFixItHints = 0;
  for (StoredDiagnostic::fixit_iterator
         FI = storedDiag.fixit_begin(),
         FE = storedDiag.fixit_end(); FI != FE; ++FI)
    DiagFixItHints[NumDiagFixItHints++] = *FI;

  assert(Client && "DiagnosticConsumer not set!");
  Level DiagLevel = storedDiag.getLevel();
  Diagnostic Info(this, storedDiag.getMessage());
  Client->HandleDiagnostic(DiagLevel, Info);
  if (Client->IncludeInDiagnosticCounts()) {
    if (DiagLevel == DiagnosticsEngine::Warning)
      ++NumWarnings;
  }

  CurDiagID = ~0U;
}

bool DiagnosticsEngine::EmitCurrentDiagnostic(bool Force) {
  assert(getClient() && "DiagnosticClient not set!");

  bool Emitted;
  if (Force) {
    Diagnostic Info(this);

    // Figure out the diagnostic level of this message.
    DiagnosticIDs::Level DiagLevel
      = Diags->getDiagnosticLevel(Info.getID(), Info.getLocation(), *this);

    Emitted = (DiagLevel != DiagnosticIDs::Ignored);
    if (Emitted) {
      // Emit the diagnostic regardless of suppression level.
      Diags->EmitDiag(*this, DiagLevel);
    }
  } else {
    // Process the diagnostic, sending the accumulated information to the
    // DiagnosticConsumer.
    Emitted = ProcessDiag();
  }

  // Clear out the current diagnostic object.
  unsigned DiagID = CurDiagID;
  Clear();

  // If there was a delayed diagnostic, emit it now.
  if (!Force && DelayedDiagID && DelayedDiagID != DiagID)
    ReportDelayed();

  return Emitted;
}


DiagnosticConsumer::~DiagnosticConsumer() {}

void DiagnosticConsumer::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                        const Diagnostic &Info) {
  if (!IncludeInDiagnosticCounts())
    return;

  if (DiagLevel == DiagnosticsEngine::Warning)
    ++NumWarnings;
  else if (DiagLevel >= DiagnosticsEngine::Error)
    ++NumErrors;
}

/// ModifierIs - Return true if the specified modifier matches specified string.
template <std::size_t StrLen>
static bool ModifierIs(const char *Modifier, unsigned ModifierLen,
                       const char (&Str)[StrLen]) {
  return StrLen-1 == ModifierLen && !memcmp(Modifier, Str, StrLen-1);
}

/// ScanForward - Scans forward, looking for the given character, skipping
/// nested clauses and escaped characters.
static const char *ScanFormat(const char *I, const char *E, char Target) {
  unsigned Depth = 0;

  for ( ; I != E; ++I) {
    if (Depth == 0 && *I == Target) return I;
    if (Depth != 0 && *I == '}') Depth--;

    if (*I == '%') {
      I++;
      if (I == E) break;

      // Escaped characters get implicitly skipped here.

      // Format specifier.
      if (!isdigit(*I) && !ispunct(*I)) {
        for (I++; I != E && !isdigit(*I) && *I != '{'; I++) ;
        if (I == E) break;
        if (*I == '{')
          Depth++;
      }
    }
  }
  return E;
}

/// HandleSelectModifier - Handle the integer 'select' modifier.  This is used
/// like this:  %select{foo|bar|baz}2.  This means that the integer argument
/// "%2" has a value from 0-2.  If the value is 0, the diagnostic prints 'foo'.
/// If the value is 1, it prints 'bar'.  If it has the value 2, it prints 'baz'.
/// This is very useful for certain classes of variant diagnostics.
static void HandleSelectModifier(const Diagnostic &DInfo, unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 SmallVectorImpl<char> &OutStr) {
  const char *ArgumentEnd = Argument+ArgumentLen;

  // Skip over 'ValNo' |'s.
  while (ValNo) {
    const char *NextVal = ScanFormat(Argument, ArgumentEnd, '|');
    assert(NextVal != ArgumentEnd && "Value for integer select modifier was"
           " larger than the number of options in the diagnostic string!");
    Argument = NextVal+1;  // Skip this string.
    --ValNo;
  }

  // Get the end of the value.  This is either the } or the |.
  const char *EndPtr = ScanFormat(Argument, ArgumentEnd, '|');

  // Recursively format the result of the select clause into the output string.
  DInfo.FormatDiagnostic(Argument, EndPtr, OutStr);
}

/// HandleIntegerSModifier - Handle the integer 's' modifier.  This adds the
/// letter 's' to the string if the value is not 1.  This is used in cases like
/// this:  "you idiot, you have %4 parameter%s4!".
static void HandleIntegerSModifier(unsigned ValNo,
                                   SmallVectorImpl<char> &OutStr) {
  if (ValNo != 1)
    OutStr.push_back('s');
}

/// HandleOrdinalModifier - Handle the integer 'ord' modifier.  This
/// prints the ordinal form of the given integer, with 1 corresponding
/// to the first ordinal.  Currently this is hard-coded to use the
/// English form.
static void HandleOrdinalModifier(unsigned ValNo,
                                  SmallVectorImpl<char> &OutStr) {
  assert(ValNo != 0 && "ValNo must be strictly positive!");

  llvm::raw_svector_ostream Out(OutStr);

  // We could use text forms for the first N ordinals, but the numeric
  // forms are actually nicer in diagnostics because they stand out.
  Out << ValNo << llvm::getOrdinalSuffix(ValNo);
}


/// PluralNumber - Parse an unsigned integer and advance Start.
static unsigned PluralNumber(const char *&Start, const char *End) {
  // Programming 101: Parse a decimal number :-)
  unsigned Val = 0;
  while (Start != End && *Start >= '0' && *Start <= '9') {
    Val *= 10;
    Val += *Start - '0';
    ++Start;
  }
  return Val;
}

/// TestPluralRange - Test if Val is in the parsed range. Modifies Start.
static bool TestPluralRange(unsigned Val, const char *&Start, const char *End) {
  if (*Start != '[') {
    unsigned Ref = PluralNumber(Start, End);
    return Ref == Val;
  }

  ++Start;
  unsigned Low = PluralNumber(Start, End);
  assert(*Start == ',' && "Bad plural expression syntax: expected ,");
  ++Start;
  unsigned High = PluralNumber(Start, End);
  assert(*Start == ']' && "Bad plural expression syntax: expected )");
  ++Start;
  return Low <= Val && Val <= High;
}

/// EvalPluralExpr - Actual expression evaluator for HandlePluralModifier.
static bool EvalPluralExpr(unsigned ValNo, const char *Start, const char *End) {
  // Empty condition?
  if (*Start == ':')
    return true;

  while (1) {
    char C = *Start;
    if (C == '%') {
      // Modulo expression
      ++Start;
      unsigned Arg = PluralNumber(Start, End);
      assert(*Start == '=' && "Bad plural expression syntax: expected =");
      ++Start;
      unsigned ValMod = ValNo % Arg;
      if (TestPluralRange(ValMod, Start, End))
        return true;
    } else {
      assert((C == '[' || (C >= '0' && C <= '9')) &&
             "Bad plural expression syntax: unexpected character");
      // Range expression
      if (TestPluralRange(ValNo, Start, End))
        return true;
    }

    // Scan for next or-expr part.
    Start = std::find(Start, End, ',');
    if (Start == End)
      break;
    ++Start;
  }
  return false;
}

/// HandlePluralModifier - Handle the integer 'plural' modifier. This is used
/// for complex plural forms, or in languages where all plurals are complex.
/// The syntax is: %plural{cond1:form1|cond2:form2|:form3}, where condn are
/// conditions that are tested in order, the form corresponding to the first
/// that applies being emitted. The empty condition is always true, making the
/// last form a default case.
/// Conditions are simple boolean expressions, where n is the number argument.
/// Here are the rules.
/// condition  := expression | empty
/// empty      :=                             -> always true
/// expression := numeric [',' expression]    -> logical or
/// numeric    := range                       -> true if n in range
///             | '%' number '=' range        -> true if n % number in range
/// range      := number
///             | '[' number ',' number ']'   -> ranges are inclusive both ends
///
/// Here are some examples from the GNU gettext manual written in this form:
/// English:
/// {1:form0|:form1}
/// Latvian:
/// {0:form2|%100=11,%10=0,%10=[2,9]:form1|:form0}
/// Gaeilge:
/// {1:form0|2:form1|:form2}
/// Romanian:
/// {1:form0|0,%100=[1,19]:form1|:form2}
/// Lithuanian:
/// {%10=0,%100=[10,19]:form2|%10=1:form0|:form1}
/// Russian (requires repeated form):
/// {%100=[11,14]:form2|%10=1:form0|%10=[2,4]:form1|:form2}
/// Slovak
/// {1:form0|[2,4]:form1|:form2}
/// Polish (requires repeated form):
/// {1:form0|%100=[10,20]:form2|%10=[2,4]:form1|:form2}
static void HandlePluralModifier(const Diagnostic &DInfo, unsigned ValNo,
                                 const char *Argument, unsigned ArgumentLen,
                                 SmallVectorImpl<char> &OutStr) {
  const char *ArgumentEnd = Argument + ArgumentLen;
  while (1) {
    assert(Argument < ArgumentEnd && "Plural expression didn't match.");
    const char *ExprEnd = Argument;
    while (*ExprEnd != ':') {
      assert(ExprEnd != ArgumentEnd && "Plural missing expression end");
      ++ExprEnd;
    }
    if (EvalPluralExpr(ValNo, Argument, ExprEnd)) {
      Argument = ExprEnd + 1;
      ExprEnd = ScanFormat(Argument, ArgumentEnd, '|');

      // Recursively format the result of the plural clause into the
      // output string.
      DInfo.FormatDiagnostic(Argument, ExprEnd, OutStr);
      return;
    }
    Argument = ScanFormat(Argument, ArgumentEnd - 1, '|') + 1;
  }
}


/// FormatDiagnostic - Format this diagnostic into a string, substituting the
/// formal arguments into the %0 slots.  The result is appended onto the Str
/// array.
void Diagnostic::
FormatDiagnostic(SmallVectorImpl<char> &OutStr) const {
  if (!StoredDiagMessage.empty()) {
    OutStr.append(StoredDiagMessage.begin(), StoredDiagMessage.end());
    return;
  }

  StringRef Diag = 
    getDiags()->getDiagnosticIDs()->getDescription(getID());

  FormatDiagnostic(Diag.begin(), Diag.end(), OutStr);
}

void Diagnostic::
FormatDiagnostic(const char *DiagStr, const char *DiagEnd,
                 SmallVectorImpl<char> &OutStr) const {

  /// FormattedArgs - Keep track of all of the arguments formatted by
  /// ConvertArgToString and pass them into subsequent calls to
  /// ConvertArgToString, allowing the implementation to avoid redundancies in
  /// obvious cases.
  SmallVector<DiagnosticsEngine::ArgumentValue, 8> FormattedArgs;

  /// QualTypeVals - Pass a vector of arrays so that QualType names can be
  /// compared to see if more information is needed to be printed.
  SmallVector<intptr_t, 2> QualTypeVals;
  SmallVector<char, 64> Tree;

  for (unsigned i = 0, e = getNumArgs(); i < e; ++i)
    if (getArgKind(i) == DiagnosticsEngine::ak_qualtype)
      QualTypeVals.push_back(getRawArg(i));

  while (DiagStr != DiagEnd) {
    if (DiagStr[0] != '%') {
      // Append non-%0 substrings to Str if we have one.
      const char *StrEnd = std::find(DiagStr, DiagEnd, '%');
      OutStr.append(DiagStr, StrEnd);
      DiagStr = StrEnd;
      continue;
    } else if (ispunct(DiagStr[1])) {
      OutStr.push_back(DiagStr[1]);  // %% -> %.
      DiagStr += 2;
      continue;
    }

    // Skip the %.
    ++DiagStr;

    // This must be a placeholder for a diagnostic argument.  The format for a
    // placeholder is one of "%0", "%modifier0", or "%modifier{arguments}0".
    // The digit is a number from 0-9 indicating which argument this comes from.
    // The modifier is a string of digits from the set [-a-z]+, arguments is a
    // brace enclosed string.
    const char *Modifier = 0, *Argument = 0;
    unsigned ModifierLen = 0, ArgumentLen = 0;

    // Check to see if we have a modifier.  If so eat it.
    if (!isdigit(DiagStr[0])) {
      Modifier = DiagStr;
      while (DiagStr[0] == '-' ||
             (DiagStr[0] >= 'a' && DiagStr[0] <= 'z'))
        ++DiagStr;
      ModifierLen = DiagStr-Modifier;

      // If we have an argument, get it next.
      if (DiagStr[0] == '{') {
        ++DiagStr; // Skip {.
        Argument = DiagStr;

        DiagStr = ScanFormat(DiagStr, DiagEnd, '}');
        assert(DiagStr != DiagEnd && "Mismatched {}'s in diagnostic string!");
        ArgumentLen = DiagStr-Argument;
        ++DiagStr;  // Skip }.
      }
    }

    assert(isdigit(*DiagStr) && "Invalid format for argument in diagnostic");
    unsigned ArgNo = *DiagStr++ - '0';

    // Only used for type diffing.
    unsigned ArgNo2 = ArgNo;

    DiagnosticsEngine::ArgumentKind Kind = getArgKind(ArgNo);
    if (Kind == DiagnosticsEngine::ak_qualtype &&
        ModifierIs(Modifier, ModifierLen, "diff")) {
      Kind = DiagnosticsEngine::ak_qualtype_pair;
      assert(*DiagStr == ',' && isdigit(*(DiagStr + 1)) &&
             "Invalid format for diff modifier");
      ++DiagStr;  // Comma.
      ArgNo2 = *DiagStr++ - '0';
      assert(getArgKind(ArgNo2) == DiagnosticsEngine::ak_qualtype &&
             "Second value of type diff must be a qualtype");
    }
    
    switch (Kind) {
    // ---- STRINGS ----
    case DiagnosticsEngine::ak_std_string: {
      const std::string &S = getArgStdStr(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");
      OutStr.append(S.begin(), S.end());
      break;
    }
    case DiagnosticsEngine::ak_c_string: {
      const char *S = getArgCStr(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");

      // Don't crash if get passed a null pointer by accident.
      if (!S)
        S = "(null)";

      OutStr.append(S, S + strlen(S));
      break;
    }
    // ---- INTEGERS ----
    case DiagnosticsEngine::ak_sint: {
      int Val = getArgSInt(ArgNo);

      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "ordinal")) {
        HandleOrdinalModifier((unsigned)Val, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        llvm::raw_svector_ostream(OutStr) << Val;
      }
      break;
    }
    case DiagnosticsEngine::ak_uint: {
      unsigned Val = getArgUInt(ArgNo);

      if (ModifierIs(Modifier, ModifierLen, "select")) {
        HandleSelectModifier(*this, Val, Argument, ArgumentLen, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "s")) {
        HandleIntegerSModifier(Val, OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "plural")) {
        HandlePluralModifier(*this, (unsigned)Val, Argument, ArgumentLen,
                             OutStr);
      } else if (ModifierIs(Modifier, ModifierLen, "ordinal")) {
        HandleOrdinalModifier(Val, OutStr);
      } else {
        assert(ModifierLen == 0 && "Unknown integer modifier");
        llvm::raw_svector_ostream(OutStr) << Val;
      }
      break;
    }
    // ---- NAMES and TYPES ----
    case DiagnosticsEngine::ak_identifierinfo: {
      const IdentifierInfo *II = getArgIdentifier(ArgNo);
      assert(ModifierLen == 0 && "No modifiers for strings yet");

      // Don't crash if get passed a null pointer by accident.
      if (!II) {
        const char *S = "(null)";
        OutStr.append(S, S + strlen(S));
        continue;
      }

      llvm::raw_svector_ostream(OutStr) << '\'' << II->getName() << '\'';
      break;
    }
    case DiagnosticsEngine::ak_qualtype:
    case DiagnosticsEngine::ak_declarationname:
    case DiagnosticsEngine::ak_nameddecl:
    case DiagnosticsEngine::ak_nestednamespec:
    case DiagnosticsEngine::ak_declcontext:
      getDiags()->ConvertArgToString(Kind, getRawArg(ArgNo),
                                     Modifier, ModifierLen,
                                     Argument, ArgumentLen,
                                     FormattedArgs.data(), FormattedArgs.size(),
                                     OutStr, QualTypeVals);
      break;
    case DiagnosticsEngine::ak_qualtype_pair:
      // Create a struct with all the info needed for printing.
      TemplateDiffTypes TDT;
      TDT.FromType = getRawArg(ArgNo);
      TDT.ToType = getRawArg(ArgNo2);
      TDT.ElideType = getDiags()->ElideType;
      TDT.ShowColors = getDiags()->ShowColors;
      TDT.TemplateDiffUsed = false;
      intptr_t val = reinterpret_cast<intptr_t>(&TDT);

      const char *ArgumentEnd = Argument + ArgumentLen;
      const char *Pipe = ScanFormat(Argument, ArgumentEnd, '|');

      // Print the tree.  If this diagnostic already has a tree, skip the
      // second tree.
      if (getDiags()->PrintTemplateTree && Tree.empty()) {
        TDT.PrintFromType = true;
        TDT.PrintTree = true;
        getDiags()->ConvertArgToString(Kind, val,
                                       Modifier, ModifierLen,
                                       Argument, ArgumentLen,
                                       FormattedArgs.data(),
                                       FormattedArgs.size(),
                                       Tree, QualTypeVals);
        // If there is no tree information, fall back to regular printing.
        if (!Tree.empty()) {
          FormatDiagnostic(Pipe + 1, ArgumentEnd, OutStr);
          break;
        }
      }

      // Non-tree printing, also the fall-back when tree printing fails.
      // The fall-back is triggered when the types compared are not templates.
      const char *FirstDollar = ScanFormat(Argument, ArgumentEnd, '$');
      const char *SecondDollar = ScanFormat(FirstDollar + 1, ArgumentEnd, '$');

      // Append before text
      FormatDiagnostic(Argument, FirstDollar, OutStr);

      // Append first type
      TDT.PrintTree = false;
      TDT.PrintFromType = true;
      getDiags()->ConvertArgToString(Kind, val,
                                     Modifier, ModifierLen,
                                     Argument, ArgumentLen,
                                     FormattedArgs.data(), FormattedArgs.size(),
                                     OutStr, QualTypeVals);
      if (!TDT.TemplateDiffUsed)
        FormattedArgs.push_back(std::make_pair(DiagnosticsEngine::ak_qualtype,
                                               TDT.FromType));

      // Append middle text
      FormatDiagnostic(FirstDollar + 1, SecondDollar, OutStr);

      // Append second type
      TDT.PrintFromType = false;
      getDiags()->ConvertArgToString(Kind, val,
                                     Modifier, ModifierLen,
                                     Argument, ArgumentLen,
                                     FormattedArgs.data(), FormattedArgs.size(),
                                     OutStr, QualTypeVals);
      if (!TDT.TemplateDiffUsed)
        FormattedArgs.push_back(std::make_pair(DiagnosticsEngine::ak_qualtype,
                                               TDT.ToType));

      // Append end text
      FormatDiagnostic(SecondDollar + 1, Pipe, OutStr);
      break;
    }
    
    // Remember this argument info for subsequent formatting operations.  Turn
    // std::strings into a null terminated string to make it be the same case as
    // all the other ones.
    if (Kind == DiagnosticsEngine::ak_qualtype_pair)
      continue;
    else if (Kind != DiagnosticsEngine::ak_std_string)
      FormattedArgs.push_back(std::make_pair(Kind, getRawArg(ArgNo)));
    else
      FormattedArgs.push_back(std::make_pair(DiagnosticsEngine::ak_c_string,
                                        (intptr_t)getArgStdStr(ArgNo).c_str()));
    
  }

  // Append the type tree to the end of the diagnostics.
  OutStr.append(Tree.begin(), Tree.end());
}

StoredDiagnostic::StoredDiagnostic() { }

StoredDiagnostic::StoredDiagnostic(DiagnosticsEngine::Level Level, unsigned ID,
                                   StringRef Message)
  : ID(ID), Level(Level), Loc(), Message(Message) { }

StoredDiagnostic::StoredDiagnostic(DiagnosticsEngine::Level Level, 
                                   const Diagnostic &Info)
  : ID(Info.getID()), Level(Level) 
{
  assert((Info.getLocation().isInvalid() || Info.hasSourceManager()) &&
       "Valid source location without setting a source manager for diagnostic");
  if (Info.getLocation().isValid())
    Loc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
  SmallString<64> Message;
  Info.FormatDiagnostic(Message);
  this->Message.assign(Message.begin(), Message.end());

  Ranges.reserve(Info.getNumRanges());
  for (unsigned I = 0, N = Info.getNumRanges(); I != N; ++I)
    Ranges.push_back(Info.getRange(I));

  FixIts.reserve(Info.getNumFixItHints());
  for (unsigned I = 0, N = Info.getNumFixItHints(); I != N; ++I)
    FixIts.push_back(Info.getFixItHint(I));
}

StoredDiagnostic::StoredDiagnostic(DiagnosticsEngine::Level Level, unsigned ID,
                                   StringRef Message, FullSourceLoc Loc,
                                   ArrayRef<CharSourceRange> Ranges,
                                   ArrayRef<FixItHint> Fixits)
  : ID(ID), Level(Level), Loc(Loc), Message(Message) 
{
  this->Ranges.assign(Ranges.begin(), Ranges.end());
  this->FixIts.assign(FixIts.begin(), FixIts.end());
}

StoredDiagnostic::~StoredDiagnostic() { }

/// IncludeInDiagnosticCounts - This method (whose default implementation
///  returns true) indicates whether the diagnostics handled by this
///  DiagnosticConsumer should be included in the number of diagnostics
///  reported by DiagnosticsEngine.
bool DiagnosticConsumer::IncludeInDiagnosticCounts() const { return true; }

void IgnoringDiagConsumer::anchor() { }

PartialDiagnostic::StorageAllocator::StorageAllocator() {
  for (unsigned I = 0; I != NumCached; ++I)
    FreeList[I] = Cached + I;
  NumFreeListEntries = NumCached;
}

PartialDiagnostic::StorageAllocator::~StorageAllocator() {
  // Don't assert if we are in a CrashRecovery context, as this invariant may
  // be invalidated during a crash.
  assert((NumFreeListEntries == NumCached || 
          llvm::CrashRecoveryContext::isRecoveringFromCrash()) && 
         "A partial is on the lamb");
}
