//===--- DiagnosticIDs.cpp - Diagnostic IDs Handling ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Diagnostic IDs-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticCategories.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <map>
using namespace clang;

//===----------------------------------------------------------------------===//
// Builtin Diagnostic information
//===----------------------------------------------------------------------===//

namespace {

// Diagnostic classes.
enum {
  CLASS_NOTE       = 0x01,
  CLASS_WARNING    = 0x02,
  CLASS_EXTENSION  = 0x03,
  CLASS_ERROR      = 0x04
};

struct StaticDiagInfoRec {
  unsigned short DiagID;
  unsigned Mapping : 3;
  unsigned Class : 3;
  unsigned SFINAE : 1;
  unsigned AccessControl : 1;
  unsigned WarnNoWerror : 1;
  unsigned WarnShowInSystemHeader : 1;
  unsigned Category : 5;

  uint16_t OptionGroupIndex;

  uint16_t DescriptionLen;
  const char *DescriptionStr;

  unsigned getOptionGroupIndex() const {
    return OptionGroupIndex;
  }

  StringRef getDescription() const {
    return StringRef(DescriptionStr, DescriptionLen);
  }

  bool operator<(const StaticDiagInfoRec &RHS) const {
    return DiagID < RHS.DiagID;
  }
};

} // namespace anonymous

static const StaticDiagInfoRec StaticDiagInfo[] = {
#define DIAG(ENUM,CLASS,DEFAULT_MAPPING,DESC,GROUP,               \
             SFINAE,ACCESS,NOWERROR,SHOWINSYSHEADER,              \
             CATEGORY)                                            \
  { diag::ENUM, DEFAULT_MAPPING, CLASS, SFINAE, ACCESS,           \
    NOWERROR, SHOWINSYSHEADER, CATEGORY, GROUP,                   \
    STR_SIZE(DESC, uint16_t), DESC },
#include "clang/Basic/DiagnosticCommonKinds.inc"
#include "clang/Basic/DiagnosticDriverKinds.inc"
#include "clang/Basic/DiagnosticFrontendKinds.inc"
#include "clang/Basic/DiagnosticSerializationKinds.inc"
#include "clang/Basic/DiagnosticLexKinds.inc"
#include "clang/Basic/DiagnosticParseKinds.inc"
#include "clang/Basic/DiagnosticASTKinds.inc"
#include "clang/Basic/DiagnosticCommentKinds.inc"
#include "clang/Basic/DiagnosticSemaKinds.inc"
#include "clang/Basic/DiagnosticAnalysisKinds.inc"
#undef DIAG
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

static const unsigned StaticDiagInfoSize =
  sizeof(StaticDiagInfo)/sizeof(StaticDiagInfo[0])-1;

/// GetDiagInfo - Return the StaticDiagInfoRec entry for the specified DiagID,
/// or null if the ID is invalid.
static const StaticDiagInfoRec *GetDiagInfo(unsigned DiagID) {
  // If assertions are enabled, verify that the StaticDiagInfo array is sorted.
#ifndef NDEBUG
  static bool IsFirst = true;
  if (IsFirst) {
    for (unsigned i = 1; i != StaticDiagInfoSize; ++i) {
      assert(StaticDiagInfo[i-1].DiagID != StaticDiagInfo[i].DiagID &&
             "Diag ID conflict, the enums at the start of clang::diag (in "
             "DiagnosticIDs.h) probably need to be increased");

      assert(StaticDiagInfo[i-1] < StaticDiagInfo[i] &&
             "Improperly sorted diag info");
    }
    IsFirst = false;
  }
#endif

  // Search the diagnostic table with a binary search.
  StaticDiagInfoRec Find = { static_cast<unsigned short>(DiagID),
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  const StaticDiagInfoRec *Found =
    std::lower_bound(StaticDiagInfo, StaticDiagInfo + StaticDiagInfoSize, Find);
  if (Found == StaticDiagInfo + StaticDiagInfoSize ||
      Found->DiagID != DiagID)
    return 0;

  return Found;
}

static DiagnosticMappingInfo GetDefaultDiagMappingInfo(unsigned DiagID) {
  DiagnosticMappingInfo Info = DiagnosticMappingInfo::Make(
    diag::MAP_FATAL, /*IsUser=*/false, /*IsPragma=*/false);

  if (const StaticDiagInfoRec *StaticInfo = GetDiagInfo(DiagID)) {
    Info.setMapping((diag::Mapping) StaticInfo->Mapping);

    if (StaticInfo->WarnNoWerror) {
      assert(Info.getMapping() == diag::MAP_WARNING &&
             "Unexpected mapping with no-Werror bit!");
      Info.setNoWarningAsError(true);
    }

    if (StaticInfo->WarnShowInSystemHeader) {
      assert(Info.getMapping() == diag::MAP_WARNING &&
             "Unexpected mapping with show-in-system-header bit!");
      Info.setShowInSystemHeader(true);
    }
  }

  return Info;
}

/// getCategoryNumberForDiag - Return the category number that a specified
/// DiagID belongs to, or 0 if no category.
unsigned DiagnosticIDs::getCategoryNumberForDiag(unsigned DiagID) {
  if (const StaticDiagInfoRec *Info = GetDiagInfo(DiagID))
    return Info->Category;
  return 0;
}

namespace {
  // The diagnostic category names.
  struct StaticDiagCategoryRec {
    const char *NameStr;
    uint8_t NameLen;

    StringRef getName() const {
      return StringRef(NameStr, NameLen);
    }
  };
}

// Unfortunately, the split between DiagnosticIDs and Diagnostic is not
// particularly clean, but for now we just implement this method here so we can
// access GetDefaultDiagMapping.
DiagnosticMappingInfo &DiagnosticsEngine::DiagState::getOrAddMappingInfo(
  diag::kind Diag)
{
  std::pair<iterator, bool> Result = DiagMap.insert(
    std::make_pair(Diag, DiagnosticMappingInfo()));

  // Initialize the entry if we added it.
  if (Result.second)
    Result.first->second = GetDefaultDiagMappingInfo(Diag);

  return Result.first->second;
}

static const StaticDiagCategoryRec CategoryNameTable[] = {
#define GET_CATEGORY_TABLE
#define CATEGORY(X, ENUM) { X, STR_SIZE(X, uint8_t) },
#include "clang/Basic/DiagnosticGroups.inc"
#undef GET_CATEGORY_TABLE
  { 0, 0 }
};

/// getNumberOfCategories - Return the number of categories
unsigned DiagnosticIDs::getNumberOfCategories() {
  return sizeof(CategoryNameTable) / sizeof(CategoryNameTable[0])-1;
}

/// getCategoryNameFromID - Given a category ID, return the name of the
/// category, an empty string if CategoryID is zero, or null if CategoryID is
/// invalid.
StringRef DiagnosticIDs::getCategoryNameFromID(unsigned CategoryID) {
  if (CategoryID >= getNumberOfCategories())
   return StringRef();
  return CategoryNameTable[CategoryID].getName();
}



DiagnosticIDs::SFINAEResponse 
DiagnosticIDs::getDiagnosticSFINAEResponse(unsigned DiagID) {
  if (const StaticDiagInfoRec *Info = GetDiagInfo(DiagID)) {
    if (Info->AccessControl)
      return SFINAE_AccessControl;
    
    if (!Info->SFINAE)
      return SFINAE_Report;

    if (Info->Class == CLASS_ERROR)
      return SFINAE_SubstitutionFailure;
    
    // Suppress notes, warnings, and extensions;
    return SFINAE_Suppress;
  }
  
  return SFINAE_Report;
}

/// getBuiltinDiagClass - Return the class field of the diagnostic.
///
static unsigned getBuiltinDiagClass(unsigned DiagID) {
  if (const StaticDiagInfoRec *Info = GetDiagInfo(DiagID))
    return Info->Class;
  return ~0U;
}

//===----------------------------------------------------------------------===//
// Custom Diagnostic information
//===----------------------------------------------------------------------===//

namespace clang {
  namespace diag {
    class CustomDiagInfo {
      typedef std::pair<DiagnosticIDs::Level, std::string> DiagDesc;
      std::vector<DiagDesc> DiagInfo;
      std::map<DiagDesc, unsigned> DiagIDs;
    public:

      /// getDescription - Return the description of the specified custom
      /// diagnostic.
      StringRef getDescription(unsigned DiagID) const {
        assert(this && DiagID-DIAG_UPPER_LIMIT < DiagInfo.size() &&
               "Invalid diagnosic ID");
        return DiagInfo[DiagID-DIAG_UPPER_LIMIT].second;
      }

      /// getLevel - Return the level of the specified custom diagnostic.
      DiagnosticIDs::Level getLevel(unsigned DiagID) const {
        assert(this && DiagID-DIAG_UPPER_LIMIT < DiagInfo.size() &&
               "Invalid diagnosic ID");
        return DiagInfo[DiagID-DIAG_UPPER_LIMIT].first;
      }

      unsigned getOrCreateDiagID(DiagnosticIDs::Level L, StringRef Message,
                                 DiagnosticIDs &Diags) {
        DiagDesc D(L, Message);
        // Check to see if it already exists.
        std::map<DiagDesc, unsigned>::iterator I = DiagIDs.lower_bound(D);
        if (I != DiagIDs.end() && I->first == D)
          return I->second;

        // If not, assign a new ID.
        unsigned ID = DiagInfo.size()+DIAG_UPPER_LIMIT;
        DiagIDs.insert(std::make_pair(D, ID));
        DiagInfo.push_back(D);
        return ID;
      }
    };

  } // end diag namespace
} // end clang namespace


//===----------------------------------------------------------------------===//
// Common Diagnostic implementation
//===----------------------------------------------------------------------===//

DiagnosticIDs::DiagnosticIDs() {
  CustomDiagInfo = 0;
}

DiagnosticIDs::~DiagnosticIDs() {
  delete CustomDiagInfo;
}

/// getCustomDiagID - Return an ID for a diagnostic with the specified message
/// and level.  If this is the first request for this diagnosic, it is
/// registered and created, otherwise the existing ID is returned.
unsigned DiagnosticIDs::getCustomDiagID(Level L, StringRef Message) {
  if (CustomDiagInfo == 0)
    CustomDiagInfo = new diag::CustomDiagInfo();
  return CustomDiagInfo->getOrCreateDiagID(L, Message, *this);
}


/// isBuiltinWarningOrExtension - Return true if the unmapped diagnostic
/// level of the specified diagnostic ID is a Warning or Extension.
/// This only works on builtin diagnostics, not custom ones, and is not legal to
/// call on NOTEs.
bool DiagnosticIDs::isBuiltinWarningOrExtension(unsigned DiagID) {
  return DiagID < diag::DIAG_UPPER_LIMIT &&
         getBuiltinDiagClass(DiagID) != CLASS_ERROR;
}

/// \brief Determine whether the given built-in diagnostic ID is a
/// Note.
bool DiagnosticIDs::isBuiltinNote(unsigned DiagID) {
  return DiagID < diag::DIAG_UPPER_LIMIT &&
    getBuiltinDiagClass(DiagID) == CLASS_NOTE;
}

/// isBuiltinExtensionDiag - Determine whether the given built-in diagnostic
/// ID is for an extension of some sort.  This also returns EnabledByDefault,
/// which is set to indicate whether the diagnostic is ignored by default (in
/// which case -pedantic enables it) or treated as a warning/error by default.
///
bool DiagnosticIDs::isBuiltinExtensionDiag(unsigned DiagID,
                                        bool &EnabledByDefault) {
  if (DiagID >= diag::DIAG_UPPER_LIMIT ||
      getBuiltinDiagClass(DiagID) != CLASS_EXTENSION)
    return false;
  
  EnabledByDefault =
    GetDefaultDiagMappingInfo(DiagID).getMapping() != diag::MAP_IGNORE;
  return true;
}

bool DiagnosticIDs::isDefaultMappingAsError(unsigned DiagID) {
  if (DiagID >= diag::DIAG_UPPER_LIMIT)
    return false;

  return GetDefaultDiagMappingInfo(DiagID).getMapping() == diag::MAP_ERROR;
}

/// getDescription - Given a diagnostic ID, return a description of the
/// issue.
StringRef DiagnosticIDs::getDescription(unsigned DiagID) const {
  if (const StaticDiagInfoRec *Info = GetDiagInfo(DiagID))
    return Info->getDescription();
  return CustomDiagInfo->getDescription(DiagID);
}

/// getDiagnosticLevel - Based on the way the client configured the
/// DiagnosticsEngine object, classify the specified diagnostic ID into a Level,
/// by consumable the DiagnosticClient.
DiagnosticIDs::Level
DiagnosticIDs::getDiagnosticLevel(unsigned DiagID, SourceLocation Loc,
                                  const DiagnosticsEngine &Diag) const {
  // Handle custom diagnostics, which cannot be mapped.
  if (DiagID >= diag::DIAG_UPPER_LIMIT)
    return CustomDiagInfo->getLevel(DiagID);

  unsigned DiagClass = getBuiltinDiagClass(DiagID);
  if (DiagClass == CLASS_NOTE) return DiagnosticIDs::Note;
  return getDiagnosticLevel(DiagID, DiagClass, Loc, Diag);
}

/// \brief Based on the way the client configured the Diagnostic
/// object, classify the specified diagnostic ID into a Level, consumable by
/// the DiagnosticClient.
///
/// \param Loc The source location we are interested in finding out the
/// diagnostic state. Can be null in order to query the latest state.
DiagnosticIDs::Level
DiagnosticIDs::getDiagnosticLevel(unsigned DiagID, unsigned DiagClass,
                                  SourceLocation Loc,
                                  const DiagnosticsEngine &Diag) const {
  // Specific non-error diagnostics may be mapped to various levels from ignored
  // to error.  Errors can only be mapped to fatal.
  DiagnosticIDs::Level Result = DiagnosticIDs::Fatal;

  DiagnosticsEngine::DiagStatePointsTy::iterator
    Pos = Diag.GetDiagStatePointForLoc(Loc);
  DiagnosticsEngine::DiagState *State = Pos->State;

  // Get the mapping information, or compute it lazily.
  DiagnosticMappingInfo &MappingInfo = State->getOrAddMappingInfo(
    (diag::kind)DiagID);

  switch (MappingInfo.getMapping()) {
  case diag::MAP_IGNORE:
    Result = DiagnosticIDs::Ignored;
    break;
  case diag::MAP_WARNING:
    Result = DiagnosticIDs::Warning;
    break;
  case diag::MAP_ERROR:
    Result = DiagnosticIDs::Error;
    break;
  case diag::MAP_FATAL:
    Result = DiagnosticIDs::Fatal;
    break;
  }

  // Upgrade ignored diagnostics if -Weverything is enabled.
  if (Diag.EnableAllWarnings && Result == DiagnosticIDs::Ignored &&
      !MappingInfo.isUser())
    Result = DiagnosticIDs::Warning;

  // Ignore -pedantic diagnostics inside __extension__ blocks.
  // (The diagnostics controlled by -pedantic are the extension diagnostics
  // that are not enabled by default.)
  bool EnabledByDefault = false;
  bool IsExtensionDiag = isBuiltinExtensionDiag(DiagID, EnabledByDefault);
  if (Diag.AllExtensionsSilenced && IsExtensionDiag && !EnabledByDefault)
    return DiagnosticIDs::Ignored;

  // For extension diagnostics that haven't been explicitly mapped, check if we
  // should upgrade the diagnostic.
  if (IsExtensionDiag && !MappingInfo.isUser()) {
    switch (Diag.ExtBehavior) {
    case DiagnosticsEngine::Ext_Ignore:
      break; 
    case DiagnosticsEngine::Ext_Warn:
      // Upgrade ignored diagnostics to warnings.
      if (Result == DiagnosticIDs::Ignored)
        Result = DiagnosticIDs::Warning;
      break;
    case DiagnosticsEngine::Ext_Error:
      // Upgrade ignored or warning diagnostics to errors.
      if (Result == DiagnosticIDs::Ignored || Result == DiagnosticIDs::Warning)
        Result = DiagnosticIDs::Error;
      break;
    }
  }

  // At this point, ignored errors can no longer be upgraded.
  if (Result == DiagnosticIDs::Ignored)
    return Result;

  // Honor -w, which is lower in priority than pedantic-errors, but higher than
  // -Werror.
  if (Result == DiagnosticIDs::Warning && Diag.IgnoreAllWarnings)
    return DiagnosticIDs::Ignored;

  // If -Werror is enabled, map warnings to errors unless explicitly disabled.
  if (Result == DiagnosticIDs::Warning) {
    if (Diag.WarningsAsErrors && !MappingInfo.hasNoWarningAsError())
      Result = DiagnosticIDs::Error;
  }

  // If -Wfatal-errors is enabled, map errors to fatal unless explicity
  // disabled.
  if (Result == DiagnosticIDs::Error) {
    if (Diag.ErrorsAsFatal && !MappingInfo.hasNoErrorAsFatal())
      Result = DiagnosticIDs::Fatal;
  }

  // If we are in a system header, we ignore it. We look at the diagnostic class
  // because we also want to ignore extensions and warnings in -Werror and
  // -pedantic-errors modes, which *map* warnings/extensions to errors.
  if (Result >= DiagnosticIDs::Warning &&
      DiagClass != CLASS_ERROR &&
      // Custom diagnostics always are emitted in system headers.
      DiagID < diag::DIAG_UPPER_LIMIT &&
      !MappingInfo.hasShowInSystemHeader() &&
      Diag.SuppressSystemWarnings &&
      Loc.isValid() &&
      Diag.getSourceManager().isInSystemHeader(
          Diag.getSourceManager().getExpansionLoc(Loc)))
    return DiagnosticIDs::Ignored;

  return Result;
}

struct clang::WarningOption {
  // Be safe with the size of 'NameLen' because we don't statically check if
  // the size will fit in the field; the struct size won't decrease with a
  // shorter type anyway.
  size_t NameLen;
  const char *NameStr;
  const short *Members;
  const short *SubGroups;

  StringRef getName() const {
    return StringRef(NameStr, NameLen);
  }
};

#define GET_DIAG_ARRAYS
#include "clang/Basic/DiagnosticGroups.inc"
#undef GET_DIAG_ARRAYS

// Second the table of options, sorted by name for fast binary lookup.
static const WarningOption OptionTable[] = {
#define GET_DIAG_TABLE
#include "clang/Basic/DiagnosticGroups.inc"
#undef GET_DIAG_TABLE
};
static const size_t OptionTableSize =
sizeof(OptionTable) / sizeof(OptionTable[0]);

static bool WarningOptionCompare(const WarningOption &LHS,
                                 const WarningOption &RHS) {
  return LHS.getName() < RHS.getName();
}

/// getWarningOptionForDiag - Return the lowest-level warning option that
/// enables the specified diagnostic.  If there is no -Wfoo flag that controls
/// the diagnostic, this returns null.
StringRef DiagnosticIDs::getWarningOptionForDiag(unsigned DiagID) {
  if (const StaticDiagInfoRec *Info = GetDiagInfo(DiagID))
    return OptionTable[Info->getOptionGroupIndex()].getName();
  return StringRef();
}

void DiagnosticIDs::getDiagnosticsInGroup(
  const WarningOption *Group,
  llvm::SmallVectorImpl<diag::kind> &Diags) const
{
  // Add the members of the option diagnostic set.
  if (const short *Member = Group->Members) {
    for (; *Member != -1; ++Member)
      Diags.push_back(*Member);
  }

  // Add the members of the subgroups.
  if (const short *SubGroups = Group->SubGroups) {
    for (; *SubGroups != (short)-1; ++SubGroups)
      getDiagnosticsInGroup(&OptionTable[(short)*SubGroups], Diags);
  }
}

bool DiagnosticIDs::getDiagnosticsInGroup(
  StringRef Group,
  llvm::SmallVectorImpl<diag::kind> &Diags) const
{
  WarningOption Key = { Group.size(), Group.data(), 0, 0 };
  const WarningOption *Found =
  std::lower_bound(OptionTable, OptionTable + OptionTableSize, Key,
                   WarningOptionCompare);
  if (Found == OptionTable + OptionTableSize ||
      Found->getName() != Group)
    return true; // Option not found.

  getDiagnosticsInGroup(Found, Diags);
  return false;
}

void DiagnosticIDs::getAllDiagnostics(
                               llvm::SmallVectorImpl<diag::kind> &Diags) const {
  for (unsigned i = 0; i != StaticDiagInfoSize; ++i)
    Diags.push_back(StaticDiagInfo[i].DiagID);
}

StringRef DiagnosticIDs::getNearestWarningOption(StringRef Group) {
  StringRef Best;
  unsigned BestDistance = Group.size() + 1; // Sanity threshold.
  for (const WarningOption *i = OptionTable, *e = OptionTable + OptionTableSize;
       i != e; ++i) {
    // Don't suggest ignored warning flags.
    if (!i->Members && !i->SubGroups)
      continue;

    unsigned Distance = i->getName().edit_distance(Group, true, BestDistance);
    if (Distance == BestDistance) {
      // Two matches with the same distance, don't prefer one over the other.
      Best = "";
    } else if (Distance < BestDistance) {
      // This is a better match.
      Best = i->getName();
      BestDistance = Distance;
    }
  }

  return Best;
}

/// ProcessDiag - This is the method used to report a diagnostic that is
/// finally fully formed.
bool DiagnosticIDs::ProcessDiag(DiagnosticsEngine &Diag) const {
  Diagnostic Info(&Diag);

  if (Diag.SuppressAllDiagnostics)
    return false;

  assert(Diag.getClient() && "DiagnosticClient not set!");

  // Figure out the diagnostic level of this message.
  unsigned DiagID = Info.getID();
  DiagnosticIDs::Level DiagLevel
    = getDiagnosticLevel(DiagID, Info.getLocation(), Diag);

  if (DiagLevel != DiagnosticIDs::Note) {
    // Record that a fatal error occurred only when we see a second
    // non-note diagnostic. This allows notes to be attached to the
    // fatal error, but suppresses any diagnostics that follow those
    // notes.
    if (Diag.LastDiagLevel == DiagnosticIDs::Fatal)
      Diag.FatalErrorOccurred = true;

    Diag.LastDiagLevel = DiagLevel;
  }

  // Update counts for DiagnosticErrorTrap even if a fatal error occurred.
  if (DiagLevel >= DiagnosticIDs::Error) {
    ++Diag.TrapNumErrorsOccurred;
    if (isUnrecoverable(DiagID))
      ++Diag.TrapNumUnrecoverableErrorsOccurred;
  }

  // If a fatal error has already been emitted, silence all subsequent
  // diagnostics.
  if (Diag.FatalErrorOccurred) {
    if (DiagLevel >= DiagnosticIDs::Error &&
        Diag.Client->IncludeInDiagnosticCounts()) {
      ++Diag.NumErrors;
      ++Diag.NumErrorsSuppressed;
    }

    return false;
  }

  // If the client doesn't care about this message, don't issue it.  If this is
  // a note and the last real diagnostic was ignored, ignore it too.
  if (DiagLevel == DiagnosticIDs::Ignored ||
      (DiagLevel == DiagnosticIDs::Note &&
       Diag.LastDiagLevel == DiagnosticIDs::Ignored))
    return false;

  if (DiagLevel >= DiagnosticIDs::Error) {
    if (isUnrecoverable(DiagID))
      Diag.UnrecoverableErrorOccurred = true;

    Diag.ErrorOccurred = true;
    if (Diag.Client->IncludeInDiagnosticCounts()) {
      ++Diag.NumErrors;
    }

    // If we've emitted a lot of errors, emit a fatal error instead of it to 
    // stop a flood of bogus errors.
    if (Diag.ErrorLimit && Diag.NumErrors > Diag.ErrorLimit &&
        DiagLevel == DiagnosticIDs::Error) {
      Diag.SetDelayedDiagnostic(diag::fatal_too_many_errors);
      return false;
    }
  }

  // Finally, report it.
  EmitDiag(Diag, DiagLevel);
  return true;
}

void DiagnosticIDs::EmitDiag(DiagnosticsEngine &Diag, Level DiagLevel) const {
  Diagnostic Info(&Diag);
  assert(DiagLevel != DiagnosticIDs::Ignored && "Cannot emit ignored diagnostics!");

  Diag.Client->HandleDiagnostic((DiagnosticsEngine::Level)DiagLevel, Info);
  if (Diag.Client->IncludeInDiagnosticCounts()) {
    if (DiagLevel == DiagnosticIDs::Warning)
      ++Diag.NumWarnings;
  }

  Diag.CurDiagID = ~0U;
}

bool DiagnosticIDs::isUnrecoverable(unsigned DiagID) const {
  if (DiagID >= diag::DIAG_UPPER_LIMIT) {
    // Custom diagnostics.
    return CustomDiagInfo->getLevel(DiagID) >= DiagnosticIDs::Error;
  }

  // Only errors may be unrecoverable.
  if (getBuiltinDiagClass(DiagID) < CLASS_ERROR)
    return false;

  if (DiagID == diag::err_unavailable ||
      DiagID == diag::err_unavailable_message)
    return false;

  // Currently we consider all ARC errors as recoverable.
  if (isARCDiagnostic(DiagID))
    return false;

  return true;
}

bool DiagnosticIDs::isARCDiagnostic(unsigned DiagID) {
  unsigned cat = getCategoryNumberForDiag(DiagID);
  return DiagnosticIDs::getCategoryNameFromID(cat).startswith("ARC ");
}
