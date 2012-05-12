//===--- ASTReader.cpp - AST File Reader ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTReader class, which reads AST files.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Serialization/ModuleManager.h"
#include "clang/Serialization/SerializationDiagnostic.h"
#include "ASTCommon.h"
#include "ASTReaderInternals.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <sys/stat.h>

using namespace clang;
using namespace clang::serialization;
using namespace clang::serialization::reader;

//===----------------------------------------------------------------------===//
// PCH validator implementation
//===----------------------------------------------------------------------===//

ASTReaderListener::~ASTReaderListener() {}

bool
PCHValidator::ReadLanguageOptions(const LangOptions &LangOpts) {
  const LangOptions &PPLangOpts = PP.getLangOpts();
  
#define LANGOPT(Name, Bits, Default, Description)         \
  if (PPLangOpts.Name != LangOpts.Name) {                 \
    Reader.Diag(diag::err_pch_langopt_mismatch)           \
      << Description << LangOpts.Name << PPLangOpts.Name; \
    return true;                                          \
  }

#define VALUE_LANGOPT(Name, Bits, Default, Description) \
  if (PPLangOpts.Name != LangOpts.Name) {               \
    Reader.Diag(diag::err_pch_langopt_value_mismatch)   \
      << Description;                                   \
  return true;                                          \
}

#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  if (PPLangOpts.get##Name() != LangOpts.get##Name()) {      \
    Reader.Diag(diag::err_pch_langopt_value_mismatch)        \
      << Description;                                        \
    return true;                                             \
  }

#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"
  
  return false;
}

bool PCHValidator::ReadTargetTriple(StringRef Triple) {
  if (Triple == PP.getTargetInfo().getTriple().str())
    return false;

  Reader.Diag(diag::warn_pch_target_triple)
    << Triple << PP.getTargetInfo().getTriple().str();
  return true;
}

namespace {
  struct EmptyStringRef {
    bool operator ()(StringRef r) const { return r.empty(); }
  };
  struct EmptyBlock {
    bool operator ()(const PCHPredefinesBlock &r) const {return r.Data.empty();}
  };
}

static bool EqualConcatenations(SmallVector<StringRef, 2> L,
                                PCHPredefinesBlocks R) {
  // First, sum up the lengths.
  unsigned LL = 0, RL = 0;
  for (unsigned I = 0, N = L.size(); I != N; ++I) {
    LL += L[I].size();
  }
  for (unsigned I = 0, N = R.size(); I != N; ++I) {
    RL += R[I].Data.size();
  }
  if (LL != RL)
    return false;
  if (LL == 0 && RL == 0)
    return true;

  // Kick out empty parts, they confuse the algorithm below.
  L.erase(std::remove_if(L.begin(), L.end(), EmptyStringRef()), L.end());
  R.erase(std::remove_if(R.begin(), R.end(), EmptyBlock()), R.end());

  // Do it the hard way. At this point, both vectors must be non-empty.
  StringRef LR = L[0], RR = R[0].Data;
  unsigned LI = 0, RI = 0, LN = L.size(), RN = R.size();
  (void) RN;
  for (;;) {
    // Compare the current pieces.
    if (LR.size() == RR.size()) {
      // If they're the same length, it's pretty easy.
      if (LR != RR)
        return false;
      // Both pieces are done, advance.
      ++LI;
      ++RI;
      // If either string is done, they're both done, since they're the same
      // length.
      if (LI == LN) {
        assert(RI == RN && "Strings not the same length after all?");
        return true;
      }
      LR = L[LI];
      RR = R[RI].Data;
    } else if (LR.size() < RR.size()) {
      // Right piece is longer.
      if (!RR.startswith(LR))
        return false;
      ++LI;
      assert(LI != LN && "Strings not the same length after all?");
      RR = RR.substr(LR.size());
      LR = L[LI];
    } else {
      // Left piece is longer.
      if (!LR.startswith(RR))
        return false;
      ++RI;
      assert(RI != RN && "Strings not the same length after all?");
      LR = LR.substr(RR.size());
      RR = R[RI].Data;
    }
  }
}

static std::pair<FileID, StringRef::size_type>
FindMacro(const PCHPredefinesBlocks &Buffers, StringRef MacroDef) {
  std::pair<FileID, StringRef::size_type> Res;
  for (unsigned I = 0, N = Buffers.size(); I != N; ++I) {
    Res.second = Buffers[I].Data.find(MacroDef);
    if (Res.second != StringRef::npos) {
      Res.first = Buffers[I].BufferID;
      break;
    }
  }
  return Res;
}

bool PCHValidator::ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                        StringRef OriginalFileName,
                                        std::string &SuggestedPredefines,
                                        FileManager &FileMgr) {
  // We are in the context of an implicit include, so the predefines buffer will
  // have a #include entry for the PCH file itself (as normalized by the
  // preprocessor initialization). Find it and skip over it in the checking
  // below.
  SmallString<256> PCHInclude;
  PCHInclude += "#include \"";
  PCHInclude += HeaderSearch::NormalizeDashIncludePath(OriginalFileName,
                                                       FileMgr);
  PCHInclude += "\"\n";
  std::pair<StringRef,StringRef> Split =
    StringRef(PP.getPredefines()).split(PCHInclude.str());
  StringRef Left =  Split.first, Right = Split.second;
  if (Left == PP.getPredefines()) {
    Error("Missing PCH include entry!");
    return true;
  }

  // If the concatenation of all the PCH buffers is equal to the adjusted
  // command line, we're done.
  SmallVector<StringRef, 2> CommandLine;
  CommandLine.push_back(Left);
  CommandLine.push_back(Right);
  if (EqualConcatenations(CommandLine, Buffers))
    return false;

  SourceManager &SourceMgr = PP.getSourceManager();

  // The predefines buffers are different. Determine what the differences are,
  // and whether they require us to reject the PCH file.
  SmallVector<StringRef, 8> PCHLines;
  for (unsigned I = 0, N = Buffers.size(); I != N; ++I)
    Buffers[I].Data.split(PCHLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  SmallVector<StringRef, 8> CmdLineLines;
  Left.split(CmdLineLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // Pick out implicit #includes after the PCH and don't consider them for
  // validation; we will insert them into SuggestedPredefines so that the
  // preprocessor includes them.
  std::string IncludesAfterPCH;
  SmallVector<StringRef, 8> AfterPCHLines;
  Right.split(AfterPCHLines, "\n", /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (unsigned i = 0, e = AfterPCHLines.size(); i != e; ++i) {
    if (AfterPCHLines[i].startswith("#include ")) {
      IncludesAfterPCH += AfterPCHLines[i];
      IncludesAfterPCH += '\n';
    } else {
      CmdLineLines.push_back(AfterPCHLines[i]);
    }
  }

  // Make sure we add the includes last into SuggestedPredefines before we
  // exit this function.
  struct AddIncludesRAII {
    std::string &SuggestedPredefines;
    std::string &IncludesAfterPCH;

    AddIncludesRAII(std::string &SuggestedPredefines,
                    std::string &IncludesAfterPCH)
      : SuggestedPredefines(SuggestedPredefines),
        IncludesAfterPCH(IncludesAfterPCH) { }
    ~AddIncludesRAII() {
      SuggestedPredefines += IncludesAfterPCH;
    }
  } AddIncludes(SuggestedPredefines, IncludesAfterPCH);

  // Sort both sets of predefined buffer lines, since we allow some extra
  // definitions and they may appear at any point in the output.
  std::sort(CmdLineLines.begin(), CmdLineLines.end());
  std::sort(PCHLines.begin(), PCHLines.end());

  // Determine which predefines that were used to build the PCH file are missing
  // from the command line.
  std::vector<StringRef> MissingPredefines;
  std::set_difference(PCHLines.begin(), PCHLines.end(),
                      CmdLineLines.begin(), CmdLineLines.end(),
                      std::back_inserter(MissingPredefines));

  bool MissingDefines = false;
  bool ConflictingDefines = false;
  for (unsigned I = 0, N = MissingPredefines.size(); I != N; ++I) {
    StringRef Missing = MissingPredefines[I];
    if (Missing.startswith("#include ")) {
      // An -include was specified when generating the PCH; it is included in
      // the PCH, just ignore it.
      continue;
    }
    if (!Missing.startswith("#define ")) {
      Reader.Diag(diag::warn_pch_compiler_options_mismatch);
      return true;
    }

    // This is a macro definition. Determine the name of the macro we're
    // defining.
    std::string::size_type StartOfMacroName = strlen("#define ");
    std::string::size_type EndOfMacroName
      = Missing.find_first_of("( \n\r", StartOfMacroName);
    assert(EndOfMacroName != std::string::npos &&
           "Couldn't find the end of the macro name");
    StringRef MacroName = Missing.slice(StartOfMacroName, EndOfMacroName);

    // Determine whether this macro was given a different definition on the
    // command line.
    std::string MacroDefStart = "#define " + MacroName.str();
    std::string::size_type MacroDefLen = MacroDefStart.size();
    SmallVector<StringRef, 8>::iterator ConflictPos
      = std::lower_bound(CmdLineLines.begin(), CmdLineLines.end(),
                         MacroDefStart);
    for (; ConflictPos != CmdLineLines.end(); ++ConflictPos) {
      if (!ConflictPos->startswith(MacroDefStart)) {
        // Different macro; we're done.
        ConflictPos = CmdLineLines.end();
        break;
      }

      assert(ConflictPos->size() > MacroDefLen &&
             "Invalid #define in predefines buffer?");
      if ((*ConflictPos)[MacroDefLen] != ' ' &&
          (*ConflictPos)[MacroDefLen] != '(')
        continue; // Longer macro name; keep trying.

      // We found a conflicting macro definition.
      break;
    }

    if (ConflictPos != CmdLineLines.end()) {
      Reader.Diag(diag::warn_cmdline_conflicting_macro_def)
          << MacroName;

      // Show the definition of this macro within the PCH file.
      std::pair<FileID, StringRef::size_type> MacroLoc =
          FindMacro(Buffers, Missing);
      assert(MacroLoc.second!=StringRef::npos && "Unable to find macro!");
      SourceLocation PCHMissingLoc =
          SourceMgr.getLocForStartOfFile(MacroLoc.first)
            .getLocWithOffset(MacroLoc.second);
      Reader.Diag(PCHMissingLoc, diag::note_pch_macro_defined_as) << MacroName;

      ConflictingDefines = true;
      continue;
    }

    // If the macro doesn't conflict, then we'll just pick up the macro
    // definition from the PCH file. Warn the user that they made a mistake.
    if (ConflictingDefines)
      continue; // Don't complain if there are already conflicting defs

    if (!MissingDefines) {
      Reader.Diag(diag::warn_cmdline_missing_macro_defs);
      MissingDefines = true;
    }

    // Show the definition of this macro within the PCH file.
    std::pair<FileID, StringRef::size_type> MacroLoc =
        FindMacro(Buffers, Missing);
    assert(MacroLoc.second!=StringRef::npos && "Unable to find macro!");
    SourceLocation PCHMissingLoc =
        SourceMgr.getLocForStartOfFile(MacroLoc.first)
          .getLocWithOffset(MacroLoc.second);
    Reader.Diag(PCHMissingLoc, diag::note_using_macro_def_from_pch);
  }

  if (ConflictingDefines)
    return true;

  // Determine what predefines were introduced based on command-line
  // parameters that were not present when building the PCH
  // file. Extra #defines are okay, so long as the identifiers being
  // defined were not used within the precompiled header.
  std::vector<StringRef> ExtraPredefines;
  std::set_difference(CmdLineLines.begin(), CmdLineLines.end(),
                      PCHLines.begin(), PCHLines.end(),
                      std::back_inserter(ExtraPredefines));
  for (unsigned I = 0, N = ExtraPredefines.size(); I != N; ++I) {
    StringRef &Extra = ExtraPredefines[I];
    if (!Extra.startswith("#define ")) {
      Reader.Diag(diag::warn_pch_compiler_options_mismatch);
      return true;
    }

    // This is an extra macro definition. Determine the name of the
    // macro we're defining.
    std::string::size_type StartOfMacroName = strlen("#define ");
    std::string::size_type EndOfMacroName
      = Extra.find_first_of("( \n\r", StartOfMacroName);
    assert(EndOfMacroName != std::string::npos &&
           "Couldn't find the end of the macro name");
    StringRef MacroName = Extra.slice(StartOfMacroName, EndOfMacroName);

    // Check whether this name was used somewhere in the PCH file. If
    // so, defining it as a macro could change behavior, so we reject
    // the PCH file.
    if (IdentifierInfo *II = Reader.get(MacroName)) {
      Reader.Diag(diag::warn_macro_name_used_in_pch) << II;
      return true;
    }

    // Add this definition to the suggested predefines buffer.
    SuggestedPredefines += Extra;
    SuggestedPredefines += '\n';
  }

  // If we get here, it's because the predefines buffer had compatible
  // contents. Accept the PCH file.
  return false;
}

void PCHValidator::ReadHeaderFileInfo(const HeaderFileInfo &HFI,
                                      unsigned ID) {
  PP.getHeaderSearchInfo().setHeaderFileInfoForUID(HFI, ID);
  ++NumHeaderInfos;
}

void PCHValidator::ReadCounter(unsigned Value) {
  PP.setCounterValue(Value);
}

//===----------------------------------------------------------------------===//
// AST reader implementation
//===----------------------------------------------------------------------===//

void
ASTReader::setDeserializationListener(ASTDeserializationListener *Listener) {
  DeserializationListener = Listener;
}



unsigned ASTSelectorLookupTrait::ComputeHash(Selector Sel) {
  return serialization::ComputeHash(Sel);
}


std::pair<unsigned, unsigned>
ASTSelectorLookupTrait::ReadKeyDataLength(const unsigned char*& d) {
  using namespace clang::io;
  unsigned KeyLen = ReadUnalignedLE16(d);
  unsigned DataLen = ReadUnalignedLE16(d);
  return std::make_pair(KeyLen, DataLen);
}

ASTSelectorLookupTrait::internal_key_type 
ASTSelectorLookupTrait::ReadKey(const unsigned char* d, unsigned) {
  using namespace clang::io;
  SelectorTable &SelTable = Reader.getContext().Selectors;
  unsigned N = ReadUnalignedLE16(d);
  IdentifierInfo *FirstII
    = Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
  if (N == 0)
    return SelTable.getNullarySelector(FirstII);
  else if (N == 1)
    return SelTable.getUnarySelector(FirstII);

  SmallVector<IdentifierInfo *, 16> Args;
  Args.push_back(FirstII);
  for (unsigned I = 1; I != N; ++I)
    Args.push_back(Reader.getLocalIdentifier(F, ReadUnalignedLE32(d)));

  return SelTable.getSelector(N, Args.data());
}

ASTSelectorLookupTrait::data_type 
ASTSelectorLookupTrait::ReadData(Selector, const unsigned char* d, 
                                 unsigned DataLen) {
  using namespace clang::io;

  data_type Result;

  Result.ID = Reader.getGlobalSelectorID(F, ReadUnalignedLE32(d));
  unsigned NumInstanceMethods = ReadUnalignedLE16(d);
  unsigned NumFactoryMethods = ReadUnalignedLE16(d);

  // Load instance methods
  for (unsigned I = 0; I != NumInstanceMethods; ++I) {
    if (ObjCMethodDecl *Method
          = Reader.GetLocalDeclAs<ObjCMethodDecl>(F, ReadUnalignedLE32(d)))
      Result.Instance.push_back(Method);
  }

  // Load factory methods
  for (unsigned I = 0; I != NumFactoryMethods; ++I) {
    if (ObjCMethodDecl *Method
          = Reader.GetLocalDeclAs<ObjCMethodDecl>(F, ReadUnalignedLE32(d)))
      Result.Factory.push_back(Method);
  }

  return Result;
}

unsigned ASTIdentifierLookupTrait::ComputeHash(const internal_key_type& a) {
  return llvm::HashString(StringRef(a.first, a.second));
}

std::pair<unsigned, unsigned>
ASTIdentifierLookupTrait::ReadKeyDataLength(const unsigned char*& d) {
  using namespace clang::io;
  unsigned DataLen = ReadUnalignedLE16(d);
  unsigned KeyLen = ReadUnalignedLE16(d);
  return std::make_pair(KeyLen, DataLen);
}

std::pair<const char*, unsigned>
ASTIdentifierLookupTrait::ReadKey(const unsigned char* d, unsigned n) {
  assert(n >= 2 && d[n-1] == '\0');
  return std::make_pair((const char*) d, n-1);
}

IdentifierInfo *ASTIdentifierLookupTrait::ReadData(const internal_key_type& k,
                                                   const unsigned char* d,
                                                   unsigned DataLen) {
  using namespace clang::io;
  unsigned RawID = ReadUnalignedLE32(d);
  bool IsInteresting = RawID & 0x01;

  // Wipe out the "is interesting" bit.
  RawID = RawID >> 1;

  IdentID ID = Reader.getGlobalIdentifierID(F, RawID);
  if (!IsInteresting) {
    // For uninteresting identifiers, just build the IdentifierInfo
    // and associate it with the persistent ID.
    IdentifierInfo *II = KnownII;
    if (!II) {
      II = &Reader.getIdentifierTable().getOwn(StringRef(k.first, k.second));
      KnownII = II;
    }
    Reader.SetIdentifierInfo(ID, II);
    II->setIsFromAST();
    Reader.markIdentifierUpToDate(II);    
    return II;
  }

  unsigned Bits = ReadUnalignedLE16(d);
  bool CPlusPlusOperatorKeyword = Bits & 0x01;
  Bits >>= 1;
  bool HasRevertedTokenIDToIdentifier = Bits & 0x01;
  Bits >>= 1;
  bool Poisoned = Bits & 0x01;
  Bits >>= 1;
  bool ExtensionToken = Bits & 0x01;
  Bits >>= 1;
  bool hasMacroDefinition = Bits & 0x01;
  Bits >>= 1;
  unsigned ObjCOrBuiltinID = Bits & 0x7FF;
  Bits >>= 11;

  assert(Bits == 0 && "Extra bits in the identifier?");
  DataLen -= 6;

  // Build the IdentifierInfo itself and link the identifier ID with
  // the new IdentifierInfo.
  IdentifierInfo *II = KnownII;
  if (!II) {
    II = &Reader.getIdentifierTable().getOwn(StringRef(k.first, k.second));
    KnownII = II;
  }
  Reader.markIdentifierUpToDate(II);
  II->setIsFromAST();

  // Set or check the various bits in the IdentifierInfo structure.
  // Token IDs are read-only.
  if (HasRevertedTokenIDToIdentifier)
    II->RevertTokenIDToIdentifier();
  II->setObjCOrBuiltinID(ObjCOrBuiltinID);
  assert(II->isExtensionToken() == ExtensionToken &&
         "Incorrect extension token flag");
  (void)ExtensionToken;
  if (Poisoned)
    II->setIsPoisoned(true);
  assert(II->isCPlusPlusOperatorKeyword() == CPlusPlusOperatorKeyword &&
         "Incorrect C++ operator keyword flag");
  (void)CPlusPlusOperatorKeyword;

  // If this identifier is a macro, deserialize the macro
  // definition.
  if (hasMacroDefinition) {
    // FIXME: Check for conflicts?
    uint32_t Offset = ReadUnalignedLE32(d);
    unsigned LocalSubmoduleID = ReadUnalignedLE32(d);
    
    // Determine whether this macro definition should be visible now, or
    // whether it is in a hidden submodule.
    bool Visible = true;
    if (SubmoduleID GlobalSubmoduleID
          = Reader.getGlobalSubmoduleID(F, LocalSubmoduleID)) {
      if (Module *Owner = Reader.getSubmodule(GlobalSubmoduleID)) {
        if (Owner->NameVisibility == Module::Hidden) {
          // The owning module is not visible, and this macro definition should
          // not be, either.
          Visible = false;
          
          // Note that this macro definition was hidden because its owning 
          // module is not yet visible.
          Reader.HiddenNamesMap[Owner].push_back(II);
        }
      } 
    }
    
    Reader.setIdentifierIsMacro(II, F, Offset, Visible);
    DataLen -= 8;
  }

  Reader.SetIdentifierInfo(ID, II);

  // Read all of the declarations visible at global scope with this
  // name.
  if (DataLen > 0) {
    SmallVector<uint32_t, 4> DeclIDs;
    for (; DataLen > 0; DataLen -= 4)
      DeclIDs.push_back(Reader.getGlobalDeclID(F, ReadUnalignedLE32(d)));
    Reader.SetGloballyVisibleDecls(II, DeclIDs);
  }

  return II;
}

unsigned 
ASTDeclContextNameLookupTrait::ComputeHash(const DeclNameKey &Key) const {
  llvm::FoldingSetNodeID ID;
  ID.AddInteger(Key.Kind);

  switch (Key.Kind) {
  case DeclarationName::Identifier:
  case DeclarationName::CXXLiteralOperatorName:
    ID.AddString(((IdentifierInfo*)Key.Data)->getName());
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    ID.AddInteger(serialization::ComputeHash(Selector(Key.Data)));
    break;
  case DeclarationName::CXXOperatorName:
    ID.AddInteger((OverloadedOperatorKind)Key.Data);
    break;
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
  case DeclarationName::CXXUsingDirective:
    break;
  }

  return ID.ComputeHash();
}

ASTDeclContextNameLookupTrait::internal_key_type 
ASTDeclContextNameLookupTrait::GetInternalKey(
                                          const external_key_type& Name) const {
  DeclNameKey Key;
  Key.Kind = Name.getNameKind();
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    Key.Data = (uint64_t)Name.getAsIdentifierInfo();
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    Key.Data = (uint64_t)Name.getObjCSelector().getAsOpaquePtr();
    break;
  case DeclarationName::CXXOperatorName:
    Key.Data = Name.getCXXOverloadedOperator();
    break;
  case DeclarationName::CXXLiteralOperatorName:
    Key.Data = (uint64_t)Name.getCXXLiteralIdentifier();
    break;
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
  case DeclarationName::CXXUsingDirective:
    Key.Data = 0;
    break;
  }

  return Key;
}

std::pair<unsigned, unsigned>
ASTDeclContextNameLookupTrait::ReadKeyDataLength(const unsigned char*& d) {
  using namespace clang::io;
  unsigned KeyLen = ReadUnalignedLE16(d);
  unsigned DataLen = ReadUnalignedLE16(d);
  return std::make_pair(KeyLen, DataLen);
}

ASTDeclContextNameLookupTrait::internal_key_type 
ASTDeclContextNameLookupTrait::ReadKey(const unsigned char* d, unsigned) {
  using namespace clang::io;

  DeclNameKey Key;
  Key.Kind = (DeclarationName::NameKind)*d++;
  switch (Key.Kind) {
  case DeclarationName::Identifier:
    Key.Data = (uint64_t)Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
    break;
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    Key.Data =
       (uint64_t)Reader.getLocalSelector(F, ReadUnalignedLE32(d))
                   .getAsOpaquePtr();
    break;
  case DeclarationName::CXXOperatorName:
    Key.Data = *d++; // OverloadedOperatorKind
    break;
  case DeclarationName::CXXLiteralOperatorName:
    Key.Data = (uint64_t)Reader.getLocalIdentifier(F, ReadUnalignedLE32(d));
    break;
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
  case DeclarationName::CXXUsingDirective:
    Key.Data = 0;
    break;
  }

  return Key;
}

ASTDeclContextNameLookupTrait::data_type 
ASTDeclContextNameLookupTrait::ReadData(internal_key_type, 
                                        const unsigned char* d,
                                        unsigned DataLen) {
  using namespace clang::io;
  unsigned NumDecls = ReadUnalignedLE16(d);
  LE32DeclID *Start = (LE32DeclID *)d;
  return std::make_pair(Start, Start + NumDecls);
}

bool ASTReader::ReadDeclContextStorage(ModuleFile &M,
                                       llvm::BitstreamCursor &Cursor,
                                   const std::pair<uint64_t, uint64_t> &Offsets,
                                       DeclContextInfo &Info) {
  SavedStreamPosition SavedPosition(Cursor);
  // First the lexical decls.
  if (Offsets.first != 0) {
    Cursor.JumpToBit(Offsets.first);

    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned Code = Cursor.ReadCode();
    unsigned RecCode = Cursor.ReadRecord(Code, Record, &Blob, &BlobLen);
    if (RecCode != DECL_CONTEXT_LEXICAL) {
      Error("Expected lexical block");
      return true;
    }

    Info.LexicalDecls = reinterpret_cast<const KindDeclIDPair*>(Blob);
    Info.NumLexicalDecls = BlobLen / sizeof(KindDeclIDPair);
  }

  // Now the lookup table.
  if (Offsets.second != 0) {
    Cursor.JumpToBit(Offsets.second);

    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned Code = Cursor.ReadCode();
    unsigned RecCode = Cursor.ReadRecord(Code, Record, &Blob, &BlobLen);
    if (RecCode != DECL_CONTEXT_VISIBLE) {
      Error("Expected visible lookup table block");
      return true;
    }
    Info.NameLookupTableData
      = ASTDeclContextNameLookupTable::Create(
                    (const unsigned char *)Blob + Record[0],
                    (const unsigned char *)Blob,
                    ASTDeclContextNameLookupTrait(*this, M));
  }

  return false;
}

void ASTReader::Error(StringRef Msg) {
  Error(diag::err_fe_pch_malformed, Msg);
}

void ASTReader::Error(unsigned DiagID,
                      StringRef Arg1, StringRef Arg2) {
  if (Diags.isDiagnosticInFlight())
    Diags.SetDelayedDiagnostic(DiagID, Arg1, Arg2);
  else
    Diag(DiagID) << Arg1 << Arg2;
}

/// \brief Tell the AST listener about the predefines buffers in the chain.
bool ASTReader::CheckPredefinesBuffers() {
  if (Listener)
    return Listener->ReadPredefinesBuffer(PCHPredefinesBuffers,
                                          ActualOriginalFileName,
                                          SuggestedPredefines,
                                          FileMgr);
  return false;
}

//===----------------------------------------------------------------------===//
// Source Manager Deserialization
//===----------------------------------------------------------------------===//

/// \brief Read the line table in the source manager block.
/// \returns true if there was an error.
bool ASTReader::ParseLineTable(ModuleFile &F,
                               SmallVectorImpl<uint64_t> &Record) {
  unsigned Idx = 0;
  LineTableInfo &LineTable = SourceMgr.getLineTable();

  // Parse the file names
  std::map<int, int> FileIDs;
  for (int I = 0, N = Record[Idx++]; I != N; ++I) {
    // Extract the file name
    unsigned FilenameLen = Record[Idx++];
    std::string Filename(&Record[Idx], &Record[Idx] + FilenameLen);
    Idx += FilenameLen;
    MaybeAddSystemRootToFilename(Filename);
    FileIDs[I] = LineTable.getLineTableFilenameID(Filename);
  }

  // Parse the line entries
  std::vector<LineEntry> Entries;
  while (Idx < Record.size()) {
    int FID = Record[Idx++];
    assert(FID >= 0 && "Serialized line entries for non-local file.");
    // Remap FileID from 1-based old view.
    FID += F.SLocEntryBaseID - 1;

    // Extract the line entries
    unsigned NumEntries = Record[Idx++];
    assert(NumEntries && "Numentries is 00000");
    Entries.clear();
    Entries.reserve(NumEntries);
    for (unsigned I = 0; I != NumEntries; ++I) {
      unsigned FileOffset = Record[Idx++];
      unsigned LineNo = Record[Idx++];
      int FilenameID = FileIDs[Record[Idx++]];
      SrcMgr::CharacteristicKind FileKind
        = (SrcMgr::CharacteristicKind)Record[Idx++];
      unsigned IncludeOffset = Record[Idx++];
      Entries.push_back(LineEntry::get(FileOffset, LineNo, FilenameID,
                                       FileKind, IncludeOffset));
    }
    LineTable.AddEntry(FID, Entries);
  }

  return false;
}

namespace {

class ASTStatData {
public:
  const ino_t ino;
  const dev_t dev;
  const mode_t mode;
  const time_t mtime;
  const off_t size;

  ASTStatData(ino_t i, dev_t d, mode_t mo, time_t m, off_t s)
    : ino(i), dev(d), mode(mo), mtime(m), size(s) {}
};

class ASTStatLookupTrait {
 public:
  typedef const char *external_key_type;
  typedef const char *internal_key_type;

  typedef ASTStatData data_type;

  static unsigned ComputeHash(const char *path) {
    return llvm::HashString(path);
  }

  static internal_key_type GetInternalKey(const char *path) { return path; }

  static bool EqualKey(internal_key_type a, internal_key_type b) {
    return strcmp(a, b) == 0;
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    unsigned KeyLen = (unsigned) clang::io::ReadUnalignedLE16(d);
    unsigned DataLen = (unsigned) *d++;
    return std::make_pair(KeyLen + 1, DataLen);
  }

  static internal_key_type ReadKey(const unsigned char *d, unsigned) {
    return (const char *)d;
  }

  static data_type ReadData(const internal_key_type, const unsigned char *d,
                            unsigned /*DataLen*/) {
    using namespace clang::io;

    ino_t ino = (ino_t) ReadUnalignedLE32(d);
    dev_t dev = (dev_t) ReadUnalignedLE32(d);
    mode_t mode = (mode_t) ReadUnalignedLE16(d);
    time_t mtime = (time_t) ReadUnalignedLE64(d);
    off_t size = (off_t) ReadUnalignedLE64(d);
    return data_type(ino, dev, mode, mtime, size);
  }
};

/// \brief stat() cache for precompiled headers.
///
/// This cache is very similar to the stat cache used by pretokenized
/// headers.
class ASTStatCache : public FileSystemStatCache {
  typedef OnDiskChainedHashTable<ASTStatLookupTrait> CacheTy;
  CacheTy *Cache;

  unsigned &NumStatHits, &NumStatMisses;
public:
  ASTStatCache(const unsigned char *Buckets, const unsigned char *Base,
               unsigned &NumStatHits, unsigned &NumStatMisses)
    : Cache(0), NumStatHits(NumStatHits), NumStatMisses(NumStatMisses) {
    Cache = CacheTy::Create(Buckets, Base);
  }

  ~ASTStatCache() { delete Cache; }

  LookupResult getStat(const char *Path, struct stat &StatBuf,
                       int *FileDescriptor) {
    // Do the lookup for the file's data in the AST file.
    CacheTy::iterator I = Cache->find(Path);

    // If we don't get a hit in the AST file just forward to 'stat'.
    if (I == Cache->end()) {
      ++NumStatMisses;
      return statChained(Path, StatBuf, FileDescriptor);
    }

    ++NumStatHits;
    ASTStatData Data = *I;

    StatBuf.st_ino = Data.ino;
    StatBuf.st_dev = Data.dev;
    StatBuf.st_mtime = Data.mtime;
    StatBuf.st_mode = Data.mode;
    StatBuf.st_size = Data.size;
    return CacheExists;
  }
};
} // end anonymous namespace


/// \brief Read a source manager block
ASTReader::ASTReadResult ASTReader::ReadSourceManagerBlock(ModuleFile &F) {
  using namespace SrcMgr;

  llvm::BitstreamCursor &SLocEntryCursor = F.SLocEntryCursor;

  // Set the source-location entry cursor to the current position in
  // the stream. This cursor will be used to read the contents of the
  // source manager block initially, and then lazily read
  // source-location entries as needed.
  SLocEntryCursor = F.Stream;

  // The stream itself is going to skip over the source manager block.
  if (F.Stream.SkipBlock()) {
    Error("malformed block record in AST file");
    return Failure;
  }

  // Enter the source manager block.
  if (SLocEntryCursor.EnterSubBlock(SOURCE_MANAGER_BLOCK_ID)) {
    Error("malformed source manager block record in AST file");
    return Failure;
  }

  RecordData Record;
  while (true) {
    unsigned Code = SLocEntryCursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (SLocEntryCursor.ReadBlockEnd()) {
        Error("error at end of Source Manager block in AST file");
        return Failure;
      }
      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      SLocEntryCursor.ReadSubBlockID();
      if (SLocEntryCursor.SkipBlock()) {
        Error("malformed block record in AST file");
        return Failure;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      SLocEntryCursor.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    const char *BlobStart;
    unsigned BlobLen;
    Record.clear();
    switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case SM_SLOC_FILE_ENTRY:
    case SM_SLOC_BUFFER_ENTRY:
    case SM_SLOC_EXPANSION_ENTRY:
      // Once we hit one of the source location entries, we're done.
      return Success;
    }
  }
}

/// \brief If a header file is not found at the path that we expect it to be
/// and the PCH file was moved from its original location, try to resolve the
/// file by assuming that header+PCH were moved together and the header is in
/// the same place relative to the PCH.
static std::string
resolveFileRelativeToOriginalDir(const std::string &Filename,
                                 const std::string &OriginalDir,
                                 const std::string &CurrDir) {
  assert(OriginalDir != CurrDir &&
         "No point trying to resolve the file if the PCH dir didn't change");
  using namespace llvm::sys;
  SmallString<128> filePath(Filename);
  fs::make_absolute(filePath);
  assert(path::is_absolute(OriginalDir));
  SmallString<128> currPCHPath(CurrDir);

  path::const_iterator fileDirI = path::begin(path::parent_path(filePath)),
                       fileDirE = path::end(path::parent_path(filePath));
  path::const_iterator origDirI = path::begin(OriginalDir),
                       origDirE = path::end(OriginalDir);
  // Skip the common path components from filePath and OriginalDir.
  while (fileDirI != fileDirE && origDirI != origDirE &&
         *fileDirI == *origDirI) {
    ++fileDirI;
    ++origDirI;
  }
  for (; origDirI != origDirE; ++origDirI)
    path::append(currPCHPath, "..");
  path::append(currPCHPath, fileDirI, fileDirE);
  path::append(currPCHPath, path::filename(Filename));
  return currPCHPath.str();
}

/// \brief Read in the source location entry with the given ID.
ASTReader::ASTReadResult ASTReader::ReadSLocEntryRecord(int ID) {
  if (ID == 0)
    return Success;

  if (unsigned(-ID) - 2 >= getTotalNumSLocs() || ID > 0) {
    Error("source location entry ID out-of-range for AST file");
    return Failure;
  }

  ModuleFile *F = GlobalSLocEntryMap.find(-ID)->second;
  F->SLocEntryCursor.JumpToBit(F->SLocEntryOffsets[ID - F->SLocEntryBaseID]);
  llvm::BitstreamCursor &SLocEntryCursor = F->SLocEntryCursor;
  unsigned BaseOffset = F->SLocEntryBaseOffset;

  ++NumSLocEntriesRead;
  unsigned Code = SLocEntryCursor.ReadCode();
  if (Code == llvm::bitc::END_BLOCK ||
      Code == llvm::bitc::ENTER_SUBBLOCK ||
      Code == llvm::bitc::DEFINE_ABBREV) {
    Error("incorrectly-formatted source location entry in AST file");
    return Failure;
  }

  RecordData Record;
  const char *BlobStart;
  unsigned BlobLen;
  switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
  default:
    Error("incorrectly-formatted source location entry in AST file");
    return Failure;

  case SM_SLOC_FILE_ENTRY: {
    if (Record.size() < 7) {
      Error("source location entry is incorrect");
      return Failure;
    }

    // We will detect whether a file changed and return 'Failure' for it, but
    // we will also try to fail gracefully by setting up the SLocEntry.
    ASTReader::ASTReadResult Result = Success;

    bool OverriddenBuffer = Record[6];
    
    std::string OrigFilename(BlobStart, BlobStart + BlobLen);
    std::string Filename = OrigFilename;
    MaybeAddSystemRootToFilename(Filename);
    const FileEntry *File = 
      OverriddenBuffer? FileMgr.getVirtualFile(Filename, (off_t)Record[4],
                                               (time_t)Record[5])
                      : FileMgr.getFile(Filename, /*OpenFile=*/false);
    if (File == 0 && !OriginalDir.empty() && !CurrentDir.empty() &&
        OriginalDir != CurrentDir) {
      std::string resolved = resolveFileRelativeToOriginalDir(Filename,
                                                              OriginalDir,
                                                              CurrentDir);
      if (!resolved.empty())
        File = FileMgr.getFile(resolved);
    }
    if (File == 0)
      File = FileMgr.getVirtualFile(Filename, (off_t)Record[4],
                                    (time_t)Record[5]);
    if (File == 0) {
      std::string ErrorStr = "could not find file '";
      ErrorStr += Filename;
      ErrorStr += "' referenced by AST file";
      Error(ErrorStr.c_str());
      return Failure;
    }

    if (!DisableValidation &&
        ((off_t)Record[4] != File->getSize()
#if !defined(LLVM_ON_WIN32)
        // In our regression testing, the Windows file system seems to
        // have inconsistent modification times that sometimes
        // erroneously trigger this error-handling path.
         || (time_t)Record[5] != File->getModificationTime()
#endif
        )) {
      Error(diag::err_fe_pch_file_modified, Filename);
      Result = Failure;
    }

    SourceLocation IncludeLoc = ReadSourceLocation(*F, Record[1]);
    if (IncludeLoc.isInvalid() && F->Kind != MK_MainFile) {
      // This is the module's main file.
      IncludeLoc = getImportLocation(F);
    }
    FileID FID = SourceMgr.createFileID(File, IncludeLoc,
                                        (SrcMgr::CharacteristicKind)Record[2],
                                        ID, BaseOffset + Record[0]);
    SrcMgr::FileInfo &FileInfo =
          const_cast<SrcMgr::FileInfo&>(SourceMgr.getSLocEntry(FID).getFile());
    FileInfo.NumCreatedFIDs = Record[7];
    if (Record[3])
      FileInfo.setHasLineDirectives();

    const DeclID *FirstDecl = F->FileSortedDecls + Record[8];
    unsigned NumFileDecls = Record[9];
    if (NumFileDecls) {
      assert(F->FileSortedDecls && "FILE_SORTED_DECLS not encountered yet ?");
      FileDeclIDs[FID] = FileDeclsInfo(F, llvm::makeArrayRef(FirstDecl,
                                                             NumFileDecls));
    }
    
    const SrcMgr::ContentCache *ContentCache
      = SourceMgr.getOrCreateContentCache(File);
    if (OverriddenBuffer && !ContentCache->BufferOverridden &&
        ContentCache->ContentsEntry == ContentCache->OrigEntry) {
      unsigned Code = SLocEntryCursor.ReadCode();
      Record.clear();
      unsigned RecCode
        = SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen);
      
      if (RecCode != SM_SLOC_BUFFER_BLOB) {
        Error("AST record has invalid code");
        return Failure;
      }
      
      llvm::MemoryBuffer *Buffer
        = llvm::MemoryBuffer::getMemBuffer(StringRef(BlobStart, BlobLen - 1),
                                           Filename);
      SourceMgr.overrideFileContents(File, Buffer);
    }

    if (Result == Failure)
      return Failure;
    break;
  }

  case SM_SLOC_BUFFER_ENTRY: {
    const char *Name = BlobStart;
    unsigned Offset = Record[0];
    unsigned Code = SLocEntryCursor.ReadCode();
    Record.clear();
    unsigned RecCode
      = SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen);

    if (RecCode != SM_SLOC_BUFFER_BLOB) {
      Error("AST record has invalid code");
      return Failure;
    }

    llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBuffer(StringRef(BlobStart, BlobLen - 1),
                                         Name);
    FileID BufferID = SourceMgr.createFileIDForMemBuffer(Buffer, ID,
                                                         BaseOffset + Offset);

    if (strcmp(Name, "<built-in>") == 0 && F->Kind == MK_PCH) {
      PCHPredefinesBlock Block = {
        BufferID,
        StringRef(BlobStart, BlobLen - 1)
      };
      PCHPredefinesBuffers.push_back(Block);
    }

    break;
  }

  case SM_SLOC_EXPANSION_ENTRY: {
    SourceLocation SpellingLoc = ReadSourceLocation(*F, Record[1]);
    SourceMgr.createExpansionLoc(SpellingLoc,
                                     ReadSourceLocation(*F, Record[2]),
                                     ReadSourceLocation(*F, Record[3]),
                                     Record[4],
                                     ID,
                                     BaseOffset + Record[0]);
    break;
  }
  }

  return Success;
}

/// \brief Find the location where the module F is imported.
SourceLocation ASTReader::getImportLocation(ModuleFile *F) {
  if (F->ImportLoc.isValid())
    return F->ImportLoc;
  
  // Otherwise we have a PCH. It's considered to be "imported" at the first
  // location of its includer.
  if (F->ImportedBy.empty() || !F->ImportedBy[0]) {
    // Main file is the importer. We assume that it is the first entry in the
    // entry table. We can't ask the manager, because at the time of PCH loading
    // the main file entry doesn't exist yet.
    // The very first entry is the invalid instantiation loc, which takes up
    // offsets 0 and 1.
    return SourceLocation::getFromRawEncoding(2U);
  }
  //return F->Loaders[0]->FirstLoc;
  return F->ImportedBy[0]->FirstLoc;
}

/// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
/// specified cursor.  Read the abbreviations that are at the top of the block
/// and then leave the cursor pointing into the block.
bool ASTReader::ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor,
                                 unsigned BlockID) {
  if (Cursor.EnterSubBlock(BlockID)) {
    Error("malformed block record in AST file");
    return Failure;
  }

  while (true) {
    uint64_t Offset = Cursor.GetCurrentBitNo();
    unsigned Code = Cursor.ReadCode();

    // We expect all abbrevs to be at the start of the block.
    if (Code != llvm::bitc::DEFINE_ABBREV) {
      Cursor.JumpToBit(Offset);
      return false;
    }
    Cursor.ReadAbbrevRecord();
  }
}

void ASTReader::ReadMacroRecord(ModuleFile &F, uint64_t Offset) {
  llvm::BitstreamCursor &Stream = F.MacroCursor;

  // Keep track of where we are in the stream, then jump back there
  // after reading this macro.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  RecordData Record;
  SmallVector<IdentifierInfo*, 16> MacroArgs;
  MacroInfo *Macro = 0;

  while (true) {
    unsigned Code = Stream.ReadCode();
    switch (Code) {
    case llvm::bitc::END_BLOCK:
      return;

    case llvm::bitc::ENTER_SUBBLOCK:
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock()) {
        Error("malformed block record in AST file");
        return;
      }
      continue;

    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    default: break;
    }

    // Read a record.
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    Record.clear();
    PreprocessorRecordTypes RecType =
      (PreprocessorRecordTypes)Stream.ReadRecord(Code, Record, BlobStart,
                                                 BlobLen);
    switch (RecType) {
    case PP_MACRO_OBJECT_LIKE:
    case PP_MACRO_FUNCTION_LIKE: {
      // If we already have a macro, that means that we've hit the end
      // of the definition of the macro we were looking for. We're
      // done.
      if (Macro)
        return;

      IdentifierInfo *II = getLocalIdentifier(F, Record[0]);
      if (II == 0) {
        Error("macro must have a name in AST file");
        return;
      }
      
      SourceLocation Loc = ReadSourceLocation(F, Record[1]);
      bool isUsed = Record[2];

      MacroInfo *MI = PP.AllocateMacroInfo(Loc);
      MI->setIsUsed(isUsed);
      MI->setIsFromAST();

      bool IsPublic = Record[3];
      unsigned NextIndex = 4;
      MI->setVisibility(IsPublic, ReadSourceLocation(F, Record, NextIndex));
      
      if (RecType == PP_MACRO_FUNCTION_LIKE) {
        // Decode function-like macro info.
        bool isC99VarArgs = Record[NextIndex++];
        bool isGNUVarArgs = Record[NextIndex++];
        MacroArgs.clear();
        unsigned NumArgs = Record[NextIndex++];
        for (unsigned i = 0; i != NumArgs; ++i)
          MacroArgs.push_back(getLocalIdentifier(F, Record[NextIndex++]));

        // Install function-like macro info.
        MI->setIsFunctionLike();
        if (isC99VarArgs) MI->setIsC99Varargs();
        if (isGNUVarArgs) MI->setIsGNUVarargs();
        MI->setArgumentList(MacroArgs.data(), MacroArgs.size(),
                            PP.getPreprocessorAllocator());
      }

      // Finally, install the macro.
      PP.setMacroInfo(II, MI, /*LoadedFromAST=*/true);

      // Remember that we saw this macro last so that we add the tokens that
      // form its body to it.
      Macro = MI;

      if (NextIndex + 1 == Record.size() && PP.getPreprocessingRecord() &&
          Record[NextIndex]) {
        // We have a macro definition. Register the association
        PreprocessedEntityID
            GlobalID = getGlobalPreprocessedEntityID(F, Record[NextIndex]);
        PreprocessingRecord &PPRec = *PP.getPreprocessingRecord();
        PPRec.RegisterMacroDefinition(Macro,
                            PPRec.getPPEntityID(GlobalID-1, /*isLoaded=*/true));
      }

      ++NumMacrosRead;
      break;
    }

    case PP_TOKEN: {
      // If we see a TOKEN before a PP_MACRO_*, then the file is
      // erroneous, just pretend we didn't see this.
      if (Macro == 0) break;

      Token Tok;
      Tok.startToken();
      Tok.setLocation(ReadSourceLocation(F, Record[0]));
      Tok.setLength(Record[1]);
      if (IdentifierInfo *II = getLocalIdentifier(F, Record[2]))
        Tok.setIdentifierInfo(II);
      Tok.setKind((tok::TokenKind)Record[3]);
      Tok.setFlag((Token::TokenFlags)Record[4]);
      Macro->AddTokenToBody(Tok);
      break;
    }
    }
  }
}

PreprocessedEntityID 
ASTReader::getGlobalPreprocessedEntityID(ModuleFile &M, unsigned LocalID) const {
  ContinuousRangeMap<uint32_t, int, 2>::const_iterator 
    I = M.PreprocessedEntityRemap.find(LocalID - NUM_PREDEF_PP_ENTITY_IDS);
  assert(I != M.PreprocessedEntityRemap.end() 
         && "Invalid index into preprocessed entity index remap");
  
  return LocalID + I->second;
}

unsigned HeaderFileInfoTrait::ComputeHash(const char *path) {
  return llvm::HashString(llvm::sys::path::filename(path));
}
    
HeaderFileInfoTrait::internal_key_type 
HeaderFileInfoTrait::GetInternalKey(const char *path) { return path; }
    
bool HeaderFileInfoTrait::EqualKey(internal_key_type a, internal_key_type b) {
  if (strcmp(a, b) == 0)
    return true;
  
  if (llvm::sys::path::filename(a) != llvm::sys::path::filename(b))
    return false;

  // Determine whether the actual files are equivalent.
  bool Result = false;
  if (llvm::sys::fs::equivalent(a, b, Result))
    return false;
  
  return Result;
}
    
std::pair<unsigned, unsigned>
HeaderFileInfoTrait::ReadKeyDataLength(const unsigned char*& d) {
  unsigned KeyLen = (unsigned) clang::io::ReadUnalignedLE16(d);
  unsigned DataLen = (unsigned) *d++;
  return std::make_pair(KeyLen + 1, DataLen);
}
    
HeaderFileInfoTrait::data_type 
HeaderFileInfoTrait::ReadData(const internal_key_type, const unsigned char *d,
                              unsigned DataLen) {
  const unsigned char *End = d + DataLen;
  using namespace clang::io;
  HeaderFileInfo HFI;
  unsigned Flags = *d++;
  HFI.isImport = (Flags >> 5) & 0x01;
  HFI.isPragmaOnce = (Flags >> 4) & 0x01;
  HFI.DirInfo = (Flags >> 2) & 0x03;
  HFI.Resolved = (Flags >> 1) & 0x01;
  HFI.IndexHeaderMapHeader = Flags & 0x01;
  HFI.NumIncludes = ReadUnalignedLE16(d);
  HFI.ControllingMacroID = Reader.getGlobalIdentifierID(M, 
                                                        ReadUnalignedLE32(d));
  if (unsigned FrameworkOffset = ReadUnalignedLE32(d)) {
    // The framework offset is 1 greater than the actual offset, 
    // since 0 is used as an indicator for "no framework name".
    StringRef FrameworkName(FrameworkStrings + FrameworkOffset - 1);
    HFI.Framework = HS->getUniqueFrameworkName(FrameworkName);
  }
  
  assert(End == d && "Wrong data length in HeaderFileInfo deserialization");
  (void)End;
        
  // This HeaderFileInfo was externally loaded.
  HFI.External = true;
  return HFI;
}

void ASTReader::setIdentifierIsMacro(IdentifierInfo *II, ModuleFile &F,
                                     uint64_t LocalOffset, bool Visible) {
  if (Visible) {
    // Note that this identifier has a macro definition.
    II->setHasMacroDefinition(true);
  }
  
  // Adjust the offset to a global offset.
  UnreadMacroRecordOffsets[II] = F.GlobalBitOffset + LocalOffset;
}

void ASTReader::ReadDefinedMacros() {
  for (ModuleReverseIterator I = ModuleMgr.rbegin(),
      E = ModuleMgr.rend(); I != E; ++I) {
    llvm::BitstreamCursor &MacroCursor = (*I)->MacroCursor;

    // If there was no preprocessor block, skip this file.
    if (!MacroCursor.getBitStreamReader())
      continue;

    llvm::BitstreamCursor Cursor = MacroCursor;
    Cursor.JumpToBit((*I)->MacroStartOffset);

    RecordData Record;
    while (true) {
      unsigned Code = Cursor.ReadCode();
      if (Code == llvm::bitc::END_BLOCK)
        break;

      if (Code == llvm::bitc::ENTER_SUBBLOCK) {
        // No known subblocks, always skip them.
        Cursor.ReadSubBlockID();
        if (Cursor.SkipBlock()) {
          Error("malformed block record in AST file");
          return;
        }
        continue;
      }

      if (Code == llvm::bitc::DEFINE_ABBREV) {
        Cursor.ReadAbbrevRecord();
        continue;
      }

      // Read a record.
      const char *BlobStart;
      unsigned BlobLen;
      Record.clear();
      switch (Cursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
      default:  // Default behavior: ignore.
        break;

      case PP_MACRO_OBJECT_LIKE:
      case PP_MACRO_FUNCTION_LIKE:
        getLocalIdentifier(**I, Record[0]);
        break;

      case PP_TOKEN:
        // Ignore tokens.
        break;
      }
    }
  }
  
  // Drain the unread macro-record offsets map.
  while (!UnreadMacroRecordOffsets.empty())
    LoadMacroDefinition(UnreadMacroRecordOffsets.begin());
}

void ASTReader::LoadMacroDefinition(
                    llvm::DenseMap<IdentifierInfo *, uint64_t>::iterator Pos) {
  assert(Pos != UnreadMacroRecordOffsets.end() && "Unknown macro definition");
  uint64_t Offset = Pos->second;
  UnreadMacroRecordOffsets.erase(Pos);
  
  RecordLocation Loc = getLocalBitOffset(Offset);
  ReadMacroRecord(*Loc.F, Loc.Offset);
}

void ASTReader::LoadMacroDefinition(IdentifierInfo *II) {
  llvm::DenseMap<IdentifierInfo *, uint64_t>::iterator Pos
    = UnreadMacroRecordOffsets.find(II);
  LoadMacroDefinition(Pos);
}

namespace {
  /// \brief Visitor class used to look up identifirs in an AST file.
  class IdentifierLookupVisitor {
    StringRef Name;
    unsigned PriorGeneration;
    IdentifierInfo *Found;
  public:
    IdentifierLookupVisitor(StringRef Name, unsigned PriorGeneration) 
      : Name(Name), PriorGeneration(PriorGeneration), Found() { }
    
    static bool visit(ModuleFile &M, void *UserData) {
      IdentifierLookupVisitor *This
        = static_cast<IdentifierLookupVisitor *>(UserData);
      
      // If we've already searched this module file, skip it now.
      if (M.Generation <= This->PriorGeneration)
        return true;
      
      ASTIdentifierLookupTable *IdTable
        = (ASTIdentifierLookupTable *)M.IdentifierLookupTable;
      if (!IdTable)
        return false;
      
      ASTIdentifierLookupTrait Trait(IdTable->getInfoObj().getReader(),
                                     M, This->Found);
                                     
      std::pair<const char*, unsigned> Key(This->Name.begin(), 
                                           This->Name.size());
      ASTIdentifierLookupTable::iterator Pos = IdTable->find(Key, &Trait);
      if (Pos == IdTable->end())
        return false;
      
      // Dereferencing the iterator has the effect of building the
      // IdentifierInfo node and populating it with the various
      // declarations it needs.
      This->Found = *Pos;
      return true;
    }
    
    // \brief Retrieve the identifier info found within the module
    // files.
    IdentifierInfo *getIdentifierInfo() const { return Found; }
  };
}

void ASTReader::updateOutOfDateIdentifier(IdentifierInfo &II) {
  unsigned PriorGeneration = 0;
  if (getContext().getLangOpts().Modules)
    PriorGeneration = IdentifierGeneration[&II];
  
  IdentifierLookupVisitor Visitor(II.getName(), PriorGeneration);
  ModuleMgr.visit(IdentifierLookupVisitor::visit, &Visitor);
  markIdentifierUpToDate(&II);
}

void ASTReader::markIdentifierUpToDate(IdentifierInfo *II) {
  if (!II)
    return;
  
  II->setOutOfDate(false);

  // Update the generation for this identifier.
  if (getContext().getLangOpts().Modules)
    IdentifierGeneration[II] = CurrentGeneration;
}

const FileEntry *ASTReader::getFileEntry(StringRef filenameStrRef) {
  std::string Filename = filenameStrRef;
  MaybeAddSystemRootToFilename(Filename);
  const FileEntry *File = FileMgr.getFile(Filename);
  if (File == 0 && !OriginalDir.empty() && !CurrentDir.empty() &&
      OriginalDir != CurrentDir) {
    std::string resolved = resolveFileRelativeToOriginalDir(Filename,
                                                            OriginalDir,
                                                            CurrentDir);
    if (!resolved.empty())
      File = FileMgr.getFile(resolved);
  }

  return File;
}

/// \brief If we are loading a relocatable PCH file, and the filename is
/// not an absolute path, add the system root to the beginning of the file
/// name.
void ASTReader::MaybeAddSystemRootToFilename(std::string &Filename) {
  // If this is not a relocatable PCH file, there's nothing to do.
  if (!RelocatablePCH)
    return;

  if (Filename.empty() || llvm::sys::path::is_absolute(Filename))
    return;

  if (isysroot.empty()) {
    // If no system root was given, default to '/'
    Filename.insert(Filename.begin(), '/');
    return;
  }

  unsigned Length = isysroot.size();
  if (isysroot[Length - 1] != '/')
    Filename.insert(Filename.begin(), '/');

  Filename.insert(Filename.begin(), isysroot.begin(), isysroot.end());
}

ASTReader::ASTReadResult
ASTReader::ReadASTBlock(ModuleFile &F) {
  llvm::BitstreamCursor &Stream = F.Stream;

  if (Stream.EnterSubBlock(AST_BLOCK_ID)) {
    Error("malformed block record in AST file");
    return Failure;
  }

  // Read all of the records and blocks for the ASt file.
  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("error at end of module block in AST file");
        return Failure;
      }

      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      case DECLTYPES_BLOCK_ID:
        // We lazily load the decls block, but we want to set up the
        // DeclsCursor cursor to point into it.  Clone our current bitcode
        // cursor to it, enter the block and read the abbrevs in that block.
        // With the main cursor, we just skip over it.
        F.DeclsCursor = Stream;
        if (Stream.SkipBlock() ||  // Skip with the main cursor.
            // Read the abbrevs.
            ReadBlockAbbrevs(F.DeclsCursor, DECLTYPES_BLOCK_ID)) {
          Error("malformed block record in AST file");
          return Failure;
        }
        break;

      case DECL_UPDATES_BLOCK_ID:
        if (Stream.SkipBlock()) {
          Error("malformed block record in AST file");
          return Failure;
        }
        break;

      case PREPROCESSOR_BLOCK_ID:
        F.MacroCursor = Stream;
        if (!PP.getExternalSource())
          PP.setExternalSource(this);

        if (Stream.SkipBlock() ||
            ReadBlockAbbrevs(F.MacroCursor, PREPROCESSOR_BLOCK_ID)) {
          Error("malformed block record in AST file");
          return Failure;
        }
        F.MacroStartOffset = F.MacroCursor.GetCurrentBitNo();
        break;

      case PREPROCESSOR_DETAIL_BLOCK_ID:
        F.PreprocessorDetailCursor = Stream;
        if (Stream.SkipBlock() ||
            ReadBlockAbbrevs(F.PreprocessorDetailCursor, 
                             PREPROCESSOR_DETAIL_BLOCK_ID)) {
          Error("malformed preprocessor detail record in AST file");
          return Failure;
        }
        F.PreprocessorDetailStartOffset
          = F.PreprocessorDetailCursor.GetCurrentBitNo();
          
        if (!PP.getPreprocessingRecord())
          PP.createPreprocessingRecord(/*RecordConditionalDirectives=*/false);
        if (!PP.getPreprocessingRecord()->getExternalSource())
          PP.getPreprocessingRecord()->SetExternalSource(*this);
        break;
        
      case SOURCE_MANAGER_BLOCK_ID:
        switch (ReadSourceManagerBlock(F)) {
        case Success:
          break;

        case Failure:
          Error("malformed source manager block in AST file");
          return Failure;

        case IgnorePCH:
          return IgnorePCH;
        }
        break;

      case SUBMODULE_BLOCK_ID:
        switch (ReadSubmoduleBlock(F)) {
        case Success:
          break;
          
        case Failure:
          Error("malformed submodule block in AST file");
          return Failure;
          
        case IgnorePCH:
          return IgnorePCH;            
        }
        break;

      default:
        if (!Stream.SkipBlock())
          break;
        Error("malformed block record in AST file");
        return Failure;
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read and process a record.
    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    switch ((ASTRecordTypes)Stream.ReadRecord(Code, Record,
                                              &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case METADATA: {
      if (Record[0] != VERSION_MAJOR && !DisableValidation) {
        Diag(Record[0] < VERSION_MAJOR? diag::warn_pch_version_too_old
                                           : diag::warn_pch_version_too_new);
        return IgnorePCH;
      }

      bool hasErrors = Record[5];
      if (hasErrors && !DisableValidation && !AllowASTWithCompilerErrors) {
        Diag(diag::err_pch_with_compiler_errors);
        return IgnorePCH;
      }

      RelocatablePCH = Record[4];
      if (Listener) {
        std::string TargetTriple(BlobStart, BlobLen);
        if (Listener->ReadTargetTriple(TargetTriple))
          return IgnorePCH;
      }
      break;
    }

    case IMPORTS: {
      // Load each of the imported PCH files. 
      unsigned Idx = 0, N = Record.size();
      while (Idx < N) {
        // Read information about the AST file.
        ModuleKind ImportedKind = (ModuleKind)Record[Idx++];
        unsigned Length = Record[Idx++];
        SmallString<128> ImportedFile(Record.begin() + Idx,
                                            Record.begin() + Idx + Length);
        Idx += Length;

        // Load the AST file.
        switch(ReadASTCore(ImportedFile, ImportedKind, &F)) {
        case Failure: return Failure;
          // If we have to ignore the dependency, we'll have to ignore this too.
        case IgnorePCH: return IgnorePCH;
        case Success: break;
        }
      }
      break;
    }

    case TYPE_OFFSET: {
      if (F.LocalNumTypes != 0) {
        Error("duplicate TYPE_OFFSET record in AST file");
        return Failure;
      }
      F.TypeOffsets = (const uint32_t *)BlobStart;
      F.LocalNumTypes = Record[0];
      unsigned LocalBaseTypeIndex = Record[1];
      F.BaseTypeIndex = getTotalNumTypes();
        
      if (F.LocalNumTypes > 0) {
        // Introduce the global -> local mapping for types within this module.
        GlobalTypeMap.insert(std::make_pair(getTotalNumTypes(), &F));
        
        // Introduce the local -> global mapping for types within this module.
        F.TypeRemap.insertOrReplace(
          std::make_pair(LocalBaseTypeIndex, 
                         F.BaseTypeIndex - LocalBaseTypeIndex));
        
        TypesLoaded.resize(TypesLoaded.size() + F.LocalNumTypes);
      }
      break;
    }
        
    case DECL_OFFSET: {
      if (F.LocalNumDecls != 0) {
        Error("duplicate DECL_OFFSET record in AST file");
        return Failure;
      }
      F.DeclOffsets = (const DeclOffset *)BlobStart;
      F.LocalNumDecls = Record[0];
      unsigned LocalBaseDeclID = Record[1];
      F.BaseDeclID = getTotalNumDecls();
        
      if (F.LocalNumDecls > 0) {
        // Introduce the global -> local mapping for declarations within this 
        // module.
        GlobalDeclMap.insert(
          std::make_pair(getTotalNumDecls() + NUM_PREDEF_DECL_IDS, &F));
        
        // Introduce the local -> global mapping for declarations within this
        // module.
        F.DeclRemap.insertOrReplace(
          std::make_pair(LocalBaseDeclID, F.BaseDeclID - LocalBaseDeclID));
        
        // Introduce the global -> local mapping for declarations within this
        // module.
        F.GlobalToLocalDeclIDs[&F] = LocalBaseDeclID;
        
        DeclsLoaded.resize(DeclsLoaded.size() + F.LocalNumDecls);
      }
      break;
    }
        
    case TU_UPDATE_LEXICAL: {
      DeclContext *TU = Context.getTranslationUnitDecl();
      DeclContextInfo &Info = F.DeclContextInfos[TU];
      Info.LexicalDecls = reinterpret_cast<const KindDeclIDPair *>(BlobStart);
      Info.NumLexicalDecls 
        = static_cast<unsigned int>(BlobLen / sizeof(KindDeclIDPair));
      TU->setHasExternalLexicalStorage(true);
      break;
    }

    case UPDATE_VISIBLE: {
      unsigned Idx = 0;
      serialization::DeclID ID = ReadDeclID(F, Record, Idx);
      ASTDeclContextNameLookupTable *Table =
        ASTDeclContextNameLookupTable::Create(
                        (const unsigned char *)BlobStart + Record[Idx++],
                        (const unsigned char *)BlobStart,
                        ASTDeclContextNameLookupTrait(*this, F));
      if (ID == PREDEF_DECL_TRANSLATION_UNIT_ID) { // Is it the TU?
        DeclContext *TU = Context.getTranslationUnitDecl();
        F.DeclContextInfos[TU].NameLookupTableData = Table;
        TU->setHasExternalVisibleStorage(true);
      } else
        PendingVisibleUpdates[ID].push_back(std::make_pair(Table, &F));
      break;
    }

    case LANGUAGE_OPTIONS:
      if (ParseLanguageOptions(Record) && !DisableValidation)
        return IgnorePCH;
      break;

    case IDENTIFIER_TABLE:
      F.IdentifierTableData = BlobStart;
      if (Record[0]) {
        F.IdentifierLookupTable
          = ASTIdentifierLookupTable::Create(
                       (const unsigned char *)F.IdentifierTableData + Record[0],
                       (const unsigned char *)F.IdentifierTableData,
                       ASTIdentifierLookupTrait(*this, F));
        
        PP.getIdentifierTable().setExternalIdentifierLookup(this);
      }
      break;

    case IDENTIFIER_OFFSET: {
      if (F.LocalNumIdentifiers != 0) {
        Error("duplicate IDENTIFIER_OFFSET record in AST file");
        return Failure;
      }
      F.IdentifierOffsets = (const uint32_t *)BlobStart;
      F.LocalNumIdentifiers = Record[0];
      unsigned LocalBaseIdentifierID = Record[1];
      F.BaseIdentifierID = getTotalNumIdentifiers();
        
      if (F.LocalNumIdentifiers > 0) {
        // Introduce the global -> local mapping for identifiers within this
        // module.
        GlobalIdentifierMap.insert(std::make_pair(getTotalNumIdentifiers() + 1, 
                                                  &F));
        
        // Introduce the local -> global mapping for identifiers within this
        // module.
        F.IdentifierRemap.insertOrReplace(
          std::make_pair(LocalBaseIdentifierID,
                         F.BaseIdentifierID - LocalBaseIdentifierID));
        
        IdentifiersLoaded.resize(IdentifiersLoaded.size() 
                                 + F.LocalNumIdentifiers);
      }
      break;
    }
        
    case EXTERNAL_DEFINITIONS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        ExternalDefinitions.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case SPECIAL_TYPES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        SpecialTypes.push_back(getGlobalTypeID(F, Record[I]));
      break;

    case STATISTICS:
      TotalNumStatements += Record[0];
      TotalNumMacros += Record[1];
      TotalLexicalDeclContexts += Record[2];
      TotalVisibleDeclContexts += Record[3];
      break;

    case UNUSED_FILESCOPED_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        UnusedFileScopedDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case DELEGATING_CTORS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        DelegatingCtorDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case WEAK_UNDECLARED_IDENTIFIERS:
      if (Record.size() % 4 != 0) {
        Error("invalid weak identifiers record");
        return Failure;
      }
        
      // FIXME: Ignore weak undeclared identifiers from non-original PCH 
      // files. This isn't the way to do it :)
      WeakUndeclaredIdentifiers.clear();
        
      // Translate the weak, undeclared identifiers into global IDs.
      for (unsigned I = 0, N = Record.size(); I < N; /* in loop */) {
        WeakUndeclaredIdentifiers.push_back(
          getGlobalIdentifierID(F, Record[I++]));
        WeakUndeclaredIdentifiers.push_back(
          getGlobalIdentifierID(F, Record[I++]));
        WeakUndeclaredIdentifiers.push_back(
          ReadSourceLocation(F, Record, I).getRawEncoding());
        WeakUndeclaredIdentifiers.push_back(Record[I++]);
      }
      break;

    case LOCALLY_SCOPED_EXTERNAL_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        LocallyScopedExternalDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case SELECTOR_OFFSETS: {
      F.SelectorOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSelectors = Record[0];
      unsigned LocalBaseSelectorID = Record[1];
      F.BaseSelectorID = getTotalNumSelectors();
        
      if (F.LocalNumSelectors > 0) {
        // Introduce the global -> local mapping for selectors within this 
        // module.
        GlobalSelectorMap.insert(std::make_pair(getTotalNumSelectors()+1, &F));
        
        // Introduce the local -> global mapping for selectors within this 
        // module.
        F.SelectorRemap.insertOrReplace(
          std::make_pair(LocalBaseSelectorID,
                         F.BaseSelectorID - LocalBaseSelectorID));

        SelectorsLoaded.resize(SelectorsLoaded.size() + F.LocalNumSelectors);        
      }
      break;
    }
        
    case METHOD_POOL:
      F.SelectorLookupTableData = (const unsigned char *)BlobStart;
      if (Record[0])
        F.SelectorLookupTable
          = ASTSelectorLookupTable::Create(
                        F.SelectorLookupTableData + Record[0],
                        F.SelectorLookupTableData,
                        ASTSelectorLookupTrait(*this, F));
      TotalNumMethodPoolEntries += Record[1];
      break;

    case REFERENCED_SELECTOR_POOL:
      if (!Record.empty()) {
        for (unsigned Idx = 0, N = Record.size() - 1; Idx < N; /* in loop */) {
          ReferencedSelectorsData.push_back(getGlobalSelectorID(F, 
                                                                Record[Idx++]));
          ReferencedSelectorsData.push_back(ReadSourceLocation(F, Record, Idx).
                                              getRawEncoding());
        }
      }
      break;

    case PP_COUNTER_VALUE:
      if (!Record.empty() && Listener)
        Listener->ReadCounter(Record[0]);
      break;
      
    case FILE_SORTED_DECLS:
      F.FileSortedDecls = (const DeclID *)BlobStart;
      break;

    case SOURCE_LOCATION_OFFSETS: {
      F.SLocEntryOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSLocEntries = Record[0];
      unsigned SLocSpaceSize = Record[1];
      llvm::tie(F.SLocEntryBaseID, F.SLocEntryBaseOffset) =
          SourceMgr.AllocateLoadedSLocEntries(F.LocalNumSLocEntries,
                                              SLocSpaceSize);
      // Make our entry in the range map. BaseID is negative and growing, so
      // we invert it. Because we invert it, though, we need the other end of
      // the range.
      unsigned RangeStart =
          unsigned(-F.SLocEntryBaseID) - F.LocalNumSLocEntries + 1;
      GlobalSLocEntryMap.insert(std::make_pair(RangeStart, &F));
      F.FirstLoc = SourceLocation::getFromRawEncoding(F.SLocEntryBaseOffset);

      // SLocEntryBaseOffset is lower than MaxLoadedOffset and decreasing.
      assert((F.SLocEntryBaseOffset & (1U << 31U)) == 0);
      GlobalSLocOffsetMap.insert(
          std::make_pair(SourceManager::MaxLoadedOffset - F.SLocEntryBaseOffset
                           - SLocSpaceSize,&F));

      // Initialize the remapping table.
      // Invalid stays invalid.
      F.SLocRemap.insert(std::make_pair(0U, 0));
      // This module. Base was 2 when being compiled.
      F.SLocRemap.insert(std::make_pair(2U,
                                  static_cast<int>(F.SLocEntryBaseOffset - 2)));
      
      TotalNumSLocEntries += F.LocalNumSLocEntries;
      break;
    }

    case MODULE_OFFSET_MAP: {
      // Additional remapping information.
      const unsigned char *Data = (const unsigned char*)BlobStart;
      const unsigned char *DataEnd = Data + BlobLen;
      
      // Continuous range maps we may be updating in our module.
      ContinuousRangeMap<uint32_t, int, 2>::Builder SLocRemap(F.SLocRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        IdentifierRemap(F.IdentifierRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        PreprocessedEntityRemap(F.PreprocessedEntityRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        SubmoduleRemap(F.SubmoduleRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder 
        SelectorRemap(F.SelectorRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder DeclRemap(F.DeclRemap);
      ContinuousRangeMap<uint32_t, int, 2>::Builder TypeRemap(F.TypeRemap);

      while(Data < DataEnd) {
        uint16_t Len = io::ReadUnalignedLE16(Data);
        StringRef Name = StringRef((const char*)Data, Len);
        Data += Len;
        ModuleFile *OM = ModuleMgr.lookup(Name);
        if (!OM) {
          Error("SourceLocation remap refers to unknown module");
          return Failure;
        }

        uint32_t SLocOffset = io::ReadUnalignedLE32(Data);
        uint32_t IdentifierIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t PreprocessedEntityIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t SubmoduleIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t SelectorIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t DeclIDOffset = io::ReadUnalignedLE32(Data);
        uint32_t TypeIndexOffset = io::ReadUnalignedLE32(Data);
        
        // Source location offset is mapped to OM->SLocEntryBaseOffset.
        SLocRemap.insert(std::make_pair(SLocOffset,
          static_cast<int>(OM->SLocEntryBaseOffset - SLocOffset)));
        IdentifierRemap.insert(
          std::make_pair(IdentifierIDOffset, 
                         OM->BaseIdentifierID - IdentifierIDOffset));
        PreprocessedEntityRemap.insert(
          std::make_pair(PreprocessedEntityIDOffset, 
            OM->BasePreprocessedEntityID - PreprocessedEntityIDOffset));
        SubmoduleRemap.insert(std::make_pair(SubmoduleIDOffset, 
                                      OM->BaseSubmoduleID - SubmoduleIDOffset));
        SelectorRemap.insert(std::make_pair(SelectorIDOffset, 
                               OM->BaseSelectorID - SelectorIDOffset));
        DeclRemap.insert(std::make_pair(DeclIDOffset, 
                                        OM->BaseDeclID - DeclIDOffset));
        
        TypeRemap.insert(std::make_pair(TypeIndexOffset, 
                                    OM->BaseTypeIndex - TypeIndexOffset));

        // Global -> local mappings.
        F.GlobalToLocalDeclIDs[OM] = DeclIDOffset;
      }
      break;
    }

    case SOURCE_MANAGER_LINE_TABLE:
      if (ParseLineTable(F, Record))
        return Failure;
      break;

    case FILE_SOURCE_LOCATION_OFFSETS:
      F.SLocFileOffsets = (const uint32_t *)BlobStart;
      F.LocalNumSLocFileEntries = Record[0];
      break;

    case SOURCE_LOCATION_PRELOADS: {
      // Need to transform from the local view (1-based IDs) to the global view,
      // which is based off F.SLocEntryBaseID.
      if (!F.PreloadSLocEntries.empty()) {
        Error("Multiple SOURCE_LOCATION_PRELOADS records in AST file");
        return Failure;
      }
      
      F.PreloadSLocEntries.swap(Record);
      break;
    }

    case STAT_CACHE: {
      if (!DisableStatCache) {
        ASTStatCache *MyStatCache =
          new ASTStatCache((const unsigned char *)BlobStart + Record[0],
                           (const unsigned char *)BlobStart,
                           NumStatHits, NumStatMisses);
        FileMgr.addStatCache(MyStatCache);
        F.StatCache = MyStatCache;
      }
      break;
    }

    case EXT_VECTOR_DECLS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        ExtVectorDecls.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case VTABLE_USES:
      if (Record.size() % 3 != 0) {
        Error("Invalid VTABLE_USES record");
        return Failure;
      }
        
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have some trouble with this. This is clearly not
      // the right way to do this.
      VTableUses.clear();
        
      for (unsigned Idx = 0, N = Record.size(); Idx != N; /* In loop */) {
        VTableUses.push_back(getGlobalDeclID(F, Record[Idx++]));
        VTableUses.push_back(
          ReadSourceLocation(F, Record, Idx).getRawEncoding());
        VTableUses.push_back(Record[Idx++]);
      }
      break;

    case DYNAMIC_CLASSES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        DynamicClasses.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case PENDING_IMPLICIT_INSTANTIATIONS:
      if (PendingInstantiations.size() % 2 != 0) {
        Error("Invalid PENDING_IMPLICIT_INSTANTIATIONS block");
        return Failure;
      }
        
      // Later lists of pending instantiations overwrite earlier ones.
      // FIXME: This is most certainly wrong for modules.
      PendingInstantiations.clear();
      for (unsigned I = 0, N = Record.size(); I != N; /* in loop */) {
        PendingInstantiations.push_back(getGlobalDeclID(F, Record[I++]));
        PendingInstantiations.push_back(
          ReadSourceLocation(F, Record, I).getRawEncoding());
      }
      break;

    case SEMA_DECL_REFS:
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have some trouble with this.
      SemaDeclRefs.clear();
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        SemaDeclRefs.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case ORIGINAL_FILE_NAME:
      // The primary AST will be the last to get here, so it will be the one
      // that's used.
      ActualOriginalFileName.assign(BlobStart, BlobLen);
      OriginalFileName = ActualOriginalFileName;
      MaybeAddSystemRootToFilename(OriginalFileName);
      break;

    case ORIGINAL_FILE_ID:
      OriginalFileID = FileID::get(Record[0]);
      break;
        
    case ORIGINAL_PCH_DIR:
      // The primary AST will be the last to get here, so it will be the one
      // that's used.
      OriginalDir.assign(BlobStart, BlobLen);
      break;

    case VERSION_CONTROL_BRANCH_REVISION: {
      const std::string &CurBranch = getClangFullRepositoryVersion();
      StringRef ASTBranch(BlobStart, BlobLen);
      if (StringRef(CurBranch) != ASTBranch && !DisableValidation) {
        Diag(diag::warn_pch_different_branch) << ASTBranch << CurBranch;
        return IgnorePCH;
      }
      break;
    }

    case PPD_ENTITIES_OFFSETS: {
      F.PreprocessedEntityOffsets = (const PPEntityOffset *)BlobStart;
      assert(BlobLen % sizeof(PPEntityOffset) == 0);
      F.NumPreprocessedEntities = BlobLen / sizeof(PPEntityOffset);

      unsigned LocalBasePreprocessedEntityID = Record[0];
      
      unsigned StartingID;
      if (!PP.getPreprocessingRecord())
        PP.createPreprocessingRecord(/*RecordConditionalDirectives=*/false);
      if (!PP.getPreprocessingRecord()->getExternalSource())
        PP.getPreprocessingRecord()->SetExternalSource(*this);
      StartingID 
        = PP.getPreprocessingRecord()
            ->allocateLoadedEntities(F.NumPreprocessedEntities);
      F.BasePreprocessedEntityID = StartingID;

      if (F.NumPreprocessedEntities > 0) {
        // Introduce the global -> local mapping for preprocessed entities in
        // this module.
        GlobalPreprocessedEntityMap.insert(std::make_pair(StartingID, &F));
       
        // Introduce the local -> global mapping for preprocessed entities in
        // this module.
        F.PreprocessedEntityRemap.insertOrReplace(
          std::make_pair(LocalBasePreprocessedEntityID,
            F.BasePreprocessedEntityID - LocalBasePreprocessedEntityID));
      }

      break;
    }
        
    case DECL_UPDATE_OFFSETS: {
      if (Record.size() % 2 != 0) {
        Error("invalid DECL_UPDATE_OFFSETS block in AST file");
        return Failure;
      }
      for (unsigned I = 0, N = Record.size(); I != N; I += 2)
        DeclUpdateOffsets[getGlobalDeclID(F, Record[I])]
          .push_back(std::make_pair(&F, Record[I+1]));
      break;
    }

    case DECL_REPLACEMENTS: {
      if (Record.size() % 3 != 0) {
        Error("invalid DECL_REPLACEMENTS block in AST file");
        return Failure;
      }
      for (unsigned I = 0, N = Record.size(); I != N; I += 3)
        ReplacedDecls[getGlobalDeclID(F, Record[I])]
          = ReplacedDeclInfo(&F, Record[I+1], Record[I+2]);
      break;
    }

    case OBJC_CATEGORIES_MAP: {
      if (F.LocalNumObjCCategoriesInMap != 0) {
        Error("duplicate OBJC_CATEGORIES_MAP record in AST file");
        return Failure;
      }
      
      F.LocalNumObjCCategoriesInMap = Record[0];
      F.ObjCCategoriesMap = (const ObjCCategoriesInfo *)BlobStart;
      break;
    }
        
    case OBJC_CATEGORIES:
      F.ObjCCategories.swap(Record);
      break;
        
    case CXX_BASE_SPECIFIER_OFFSETS: {
      if (F.LocalNumCXXBaseSpecifiers != 0) {
        Error("duplicate CXX_BASE_SPECIFIER_OFFSETS record in AST file");
        return Failure;
      }
      
      F.LocalNumCXXBaseSpecifiers = Record[0];
      F.CXXBaseSpecifiersOffsets = (const uint32_t *)BlobStart;
      NumCXXBaseSpecifiersLoaded += F.LocalNumCXXBaseSpecifiers;
      break;
    }

    case DIAG_PRAGMA_MAPPINGS:
      if (Record.size() % 2 != 0) {
        Error("invalid DIAG_USER_MAPPINGS block in AST file");
        return Failure;
      }
        
      if (F.PragmaDiagMappings.empty())
        F.PragmaDiagMappings.swap(Record);
      else
        F.PragmaDiagMappings.insert(F.PragmaDiagMappings.end(),
                                    Record.begin(), Record.end());
      break;
        
    case CUDA_SPECIAL_DECL_REFS:
      // Later tables overwrite earlier ones.
      // FIXME: Modules will have trouble with this.
      CUDASpecialDeclRefs.clear();
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        CUDASpecialDeclRefs.push_back(getGlobalDeclID(F, Record[I]));
      break;

    case HEADER_SEARCH_TABLE: {
      F.HeaderFileInfoTableData = BlobStart;
      F.LocalNumHeaderFileInfos = Record[1];
      F.HeaderFileFrameworkStrings = BlobStart + Record[2];
      if (Record[0]) {
        F.HeaderFileInfoTable
          = HeaderFileInfoLookupTable::Create(
                   (const unsigned char *)F.HeaderFileInfoTableData + Record[0],
                   (const unsigned char *)F.HeaderFileInfoTableData,
                   HeaderFileInfoTrait(*this, F, 
                                       &PP.getHeaderSearchInfo(),
                                       BlobStart + Record[2]));
        
        PP.getHeaderSearchInfo().SetExternalSource(this);
        if (!PP.getHeaderSearchInfo().getExternalLookup())
          PP.getHeaderSearchInfo().SetExternalLookup(this);
      }
      break;
    }
        
    case FP_PRAGMA_OPTIONS:
      // Later tables overwrite earlier ones.
      FPPragmaOptions.swap(Record);
      break;

    case OPENCL_EXTENSIONS:
      // Later tables overwrite earlier ones.
      OpenCLExtensions.swap(Record);
      break;

    case TENTATIVE_DEFINITIONS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        TentativeDefinitions.push_back(getGlobalDeclID(F, Record[I]));
      break;
        
    case KNOWN_NAMESPACES:
      for (unsigned I = 0, N = Record.size(); I != N; ++I)
        KnownNamespaces.push_back(getGlobalDeclID(F, Record[I]));
      break;
        
    case IMPORTED_MODULES: {
      if (F.Kind != MK_Module) {
        // If we aren't loading a module (which has its own exports), make
        // all of the imported modules visible.
        // FIXME: Deal with macros-only imports.
        for (unsigned I = 0, N = Record.size(); I != N; ++I) {
          if (unsigned GlobalID = getGlobalSubmoduleID(F, Record[I]))
            ImportedModules.push_back(GlobalID);
        }
      }
      break;
    }

    case LOCAL_REDECLARATIONS: {
      F.RedeclarationChains.swap(Record);
      break;
    }
        
    case LOCAL_REDECLARATIONS_MAP: {
      if (F.LocalNumRedeclarationsInMap != 0) {
        Error("duplicate LOCAL_REDECLARATIONS_MAP record in AST file");
        return Failure;
      }
      
      F.LocalNumRedeclarationsInMap = Record[0];
      F.RedeclarationsMap = (const LocalRedeclarationsInfo *)BlobStart;
      break;
    }
        
    case MERGED_DECLARATIONS: {
      for (unsigned Idx = 0; Idx < Record.size(); /* increment in loop */) {
        GlobalDeclID CanonID = getGlobalDeclID(F, Record[Idx++]);
        SmallVectorImpl<GlobalDeclID> &Decls = StoredMergedDecls[CanonID];
        for (unsigned N = Record[Idx++]; N > 0; --N)
          Decls.push_back(getGlobalDeclID(F, Record[Idx++]));
      }
      break;
    }
    }
  }
  Error("premature end of bitstream in AST file");
  return Failure;
}

ASTReader::ASTReadResult ASTReader::validateFileEntries(ModuleFile &M) {
  llvm::BitstreamCursor &SLocEntryCursor = M.SLocEntryCursor;

  for (unsigned i = 0, e = M.LocalNumSLocFileEntries; i != e; ++i) {
    SLocEntryCursor.JumpToBit(M.SLocFileOffsets[i]);
    unsigned Code = SLocEntryCursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK ||
        Code == llvm::bitc::ENTER_SUBBLOCK ||
        Code == llvm::bitc::DEFINE_ABBREV) {
      Error("incorrectly-formatted source location entry in AST file");
      return Failure;
    }

    RecordData Record;
    const char *BlobStart;
    unsigned BlobLen;
    switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
    default:
      Error("incorrectly-formatted source location entry in AST file");
      return Failure;

    case SM_SLOC_FILE_ENTRY: {
      // If the buffer was overridden, the file need not exist.
      if (Record[6])
        break;
      
      StringRef Filename(BlobStart, BlobLen);
      const FileEntry *File = getFileEntry(Filename);

      if (File == 0) {
        std::string ErrorStr = "could not find file '";
        ErrorStr += Filename;
        ErrorStr += "' referenced by AST file";
        Error(ErrorStr.c_str());
        return IgnorePCH;
      }

      if (Record.size() < 7) {
        Error("source location entry is incorrect");
        return Failure;
      }

      // The stat info from the FileEntry came from the cached stat
      // info of the PCH, so we cannot trust it.
      struct stat StatBuf;
      if (::stat(File->getName(), &StatBuf) != 0) {
        StatBuf.st_size = File->getSize();
        StatBuf.st_mtime = File->getModificationTime();
      }

      if (((off_t)Record[4] != StatBuf.st_size
#if !defined(LLVM_ON_WIN32)
          // In our regression testing, the Windows file system seems to
          // have inconsistent modification times that sometimes
          // erroneously trigger this error-handling path.
           || (time_t)Record[5] != StatBuf.st_mtime
#endif
          )) {
        Error(diag::err_fe_pch_file_modified, Filename);
        return IgnorePCH;
      }

      break;
    }
    }
  }

  return Success;
}

void ASTReader::makeNamesVisible(const HiddenNames &Names) {
  for (unsigned I = 0, N = Names.size(); I != N; ++I) {
    if (Decl *D = Names[I].dyn_cast<Decl *>())
      D->Hidden = false;
    else {
      IdentifierInfo *II = Names[I].get<IdentifierInfo *>();
      if (!II->hasMacroDefinition()) {
        II->setHasMacroDefinition(true);
        if (DeserializationListener)
          DeserializationListener->MacroVisible(II);
      }
    }
  }
}

void ASTReader::makeModuleVisible(Module *Mod, 
                                  Module::NameVisibilityKind NameVisibility) {
  llvm::SmallPtrSet<Module *, 4> Visited;
  llvm::SmallVector<Module *, 4> Stack;
  Stack.push_back(Mod);  
  while (!Stack.empty()) {
    Mod = Stack.back();
    Stack.pop_back();

    if (NameVisibility <= Mod->NameVisibility) {
      // This module already has this level of visibility (or greater), so 
      // there is nothing more to do.
      continue;
    }
    
    if (!Mod->isAvailable()) {
      // Modules that aren't available cannot be made visible.
      continue;
    }

    // Update the module's name visibility.
    Mod->NameVisibility = NameVisibility;
    
    // If we've already deserialized any names from this module,
    // mark them as visible.
    HiddenNamesMapType::iterator Hidden = HiddenNamesMap.find(Mod);
    if (Hidden != HiddenNamesMap.end()) {
      makeNamesVisible(Hidden->second);
      HiddenNamesMap.erase(Hidden);
    }
    
    // Push any non-explicit submodules onto the stack to be marked as
    // visible.
    for (Module::submodule_iterator Sub = Mod->submodule_begin(),
                                 SubEnd = Mod->submodule_end();
         Sub != SubEnd; ++Sub) {
      if (!(*Sub)->IsExplicit && Visited.insert(*Sub))
        Stack.push_back(*Sub);
    }
    
    // Push any exported modules onto the stack to be marked as visible.
    bool AnyWildcard = false;
    bool UnrestrictedWildcard = false;
    llvm::SmallVector<Module *, 4> WildcardRestrictions;
    for (unsigned I = 0, N = Mod->Exports.size(); I != N; ++I) {
      Module *Exported = Mod->Exports[I].getPointer();
      if (!Mod->Exports[I].getInt()) {
        // Export a named module directly; no wildcards involved.
        if (Visited.insert(Exported))
          Stack.push_back(Exported);
        
        continue;
      }
      
      // Wildcard export: export all of the imported modules that match
      // the given pattern.
      AnyWildcard = true;
      if (UnrestrictedWildcard)
        continue;

      if (Module *Restriction = Mod->Exports[I].getPointer())
        WildcardRestrictions.push_back(Restriction);
      else {
        WildcardRestrictions.clear();
        UnrestrictedWildcard = true;
      }
    }
    
    // If there were any wildcards, push any imported modules that were
    // re-exported by the wildcard restriction.
    if (!AnyWildcard)
      continue;
    
    for (unsigned I = 0, N = Mod->Imports.size(); I != N; ++I) {
      Module *Imported = Mod->Imports[I];
      if (Visited.count(Imported))
        continue;
      
      bool Acceptable = UnrestrictedWildcard;
      if (!Acceptable) {
        // Check whether this module meets one of the restrictions.
        for (unsigned R = 0, NR = WildcardRestrictions.size(); R != NR; ++R) {
          Module *Restriction = WildcardRestrictions[R];
          if (Imported == Restriction || Imported->isSubModuleOf(Restriction)) {
            Acceptable = true;
            break;
          }
        }
      }
      
      if (!Acceptable)
        continue;
      
      Visited.insert(Imported);
      Stack.push_back(Imported);
    }
  }
}

ASTReader::ASTReadResult ASTReader::ReadAST(const std::string &FileName,
                                            ModuleKind Type) {
  // Bump the generation number.
  unsigned PreviousGeneration = CurrentGeneration++;
  
  switch(ReadASTCore(FileName, Type, /*ImportedBy=*/0)) {
  case Failure: return Failure;
  case IgnorePCH: return IgnorePCH;
  case Success: break;
  }

  // Here comes stuff that we only do once the entire chain is loaded.

  // Check the predefines buffers.
  if (!DisableValidation && Type == MK_PCH &&
      // FIXME: CheckPredefinesBuffers also sets the SuggestedPredefines;
      // if DisableValidation is true, defines that were set on command-line
      // but not in the PCH file will not be added to SuggestedPredefines.
      CheckPredefinesBuffers())
    return IgnorePCH;

  // Mark all of the identifiers in the identifier table as being out of date,
  // so that various accessors know to check the loaded modules when the
  // identifier is used.
  for (IdentifierTable::iterator Id = PP.getIdentifierTable().begin(),
                              IdEnd = PP.getIdentifierTable().end();
       Id != IdEnd; ++Id)
    Id->second->setOutOfDate(true);
  
  // Resolve any unresolved module exports.
  for (unsigned I = 0, N = UnresolvedModuleImportExports.size(); I != N; ++I) {
    UnresolvedModuleImportExport &Unresolved = UnresolvedModuleImportExports[I];
    SubmoduleID GlobalID = getGlobalSubmoduleID(*Unresolved.File,Unresolved.ID);
    Module *ResolvedMod = getSubmodule(GlobalID);
    
    if (Unresolved.IsImport) {
      if (ResolvedMod)
        Unresolved.Mod->Imports.push_back(ResolvedMod);
      continue;
    }

    if (ResolvedMod || Unresolved.IsWildcard)
      Unresolved.Mod->Exports.push_back(
        Module::ExportDecl(ResolvedMod, Unresolved.IsWildcard));
  }
  UnresolvedModuleImportExports.clear();
  
  InitializeContext();

  if (DeserializationListener)
    DeserializationListener->ReaderInitialized(this);

  if (!OriginalFileID.isInvalid()) {
    OriginalFileID = FileID::get(ModuleMgr.getPrimaryModule().SLocEntryBaseID
                                      + OriginalFileID.getOpaqueValue() - 1);

    // If this AST file is a precompiled preamble, then set the preamble file ID
    // of the source manager to the file source file from which the preamble was
    // built.
    if (Type == MK_Preamble) {
      SourceMgr.setPreambleFileID(OriginalFileID);
    } else if (Type == MK_MainFile) {
      SourceMgr.setMainFileID(OriginalFileID);
    }
  }
  
  // For any Objective-C class definitions we have already loaded, make sure
  // that we load any additional categories.
  for (unsigned I = 0, N = ObjCClassesLoaded.size(); I != N; ++I) {
    loadObjCCategories(ObjCClassesLoaded[I]->getGlobalID(), 
                       ObjCClassesLoaded[I],
                       PreviousGeneration);
  }
  
  return Success;
}

ASTReader::ASTReadResult ASTReader::ReadASTCore(StringRef FileName,
                                                ModuleKind Type,
                                                ModuleFile *ImportedBy) {
  ModuleFile *M;
  bool NewModule;
  std::string ErrorStr;
  llvm::tie(M, NewModule) = ModuleMgr.addModule(FileName, Type, ImportedBy,
                                                CurrentGeneration, ErrorStr);

  if (!M) {
    // We couldn't load the module.
    std::string Msg = "Unable to load module \"" + FileName.str() + "\": "
      + ErrorStr;
    Error(Msg);
    return Failure;
  }

  if (!NewModule) {
    // We've already loaded this module.
    return Success;
  }

  // FIXME: This seems rather a hack. Should CurrentDir be part of the
  // module?
  if (FileName != "-") {
    CurrentDir = llvm::sys::path::parent_path(FileName);
    if (CurrentDir.empty()) CurrentDir = ".";
  }

  ModuleFile &F = *M;
  llvm::BitstreamCursor &Stream = F.Stream;
  Stream.init(F.StreamFile);
  F.SizeInBits = F.Buffer->getBufferSize() * 8;
  
  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H') {
    Diag(diag::err_not_a_pch_file) << FileName;
    return Failure;
  }

  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    if (Code != llvm::bitc::ENTER_SUBBLOCK) {
      Error("invalid record at top-level of AST file");
      return Failure;
    }

    unsigned BlockID = Stream.ReadSubBlockID();

    // We only know the AST subblock ID.
    switch (BlockID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock()) {
        Error("malformed BlockInfoBlock in AST file");
        return Failure;
      }
      break;
    case AST_BLOCK_ID:
      switch (ReadASTBlock(F)) {
      case Success:
        break;

      case Failure:
        return Failure;

      case IgnorePCH:
        // FIXME: We could consider reading through to the end of this
        // AST block, skipping subblocks, to see if there are other
        // AST blocks elsewhere.

        // FIXME: We can't clear loaded slocentries anymore.
        //SourceMgr.ClearPreallocatedSLocEntries();

        // Remove the stat cache.
        if (F.StatCache)
          FileMgr.removeStatCache((ASTStatCache*)F.StatCache);

        return IgnorePCH;
      }
      break;
    default:
      if (Stream.SkipBlock()) {
        Error("malformed block record in AST file");
        return Failure;
      }
      break;
    }
  }
  
  // Once read, set the ModuleFile bit base offset and update the size in 
  // bits of all files we've seen.
  F.GlobalBitOffset = TotalModulesSizeInBits;
  TotalModulesSizeInBits += F.SizeInBits;
  GlobalBitOffsetsMap.insert(std::make_pair(F.GlobalBitOffset, &F));

  // Make sure that the files this module was built against are still available.
  if (!DisableValidation) {
    switch(validateFileEntries(*M)) {
    case Failure: return Failure;
    case IgnorePCH: return IgnorePCH;
    case Success: break;
    }
  }
  
  // Preload SLocEntries.
  for (unsigned I = 0, N = M->PreloadSLocEntries.size(); I != N; ++I) {
    int Index = int(M->PreloadSLocEntries[I] - 1) + F.SLocEntryBaseID;
    // Load it through the SourceManager and don't call ReadSLocEntryRecord()
    // directly because the entry may have already been loaded in which case
    // calling ReadSLocEntryRecord() directly would trigger an assertion in
    // SourceManager.
    SourceMgr.getLoadedSLocEntryByID(Index);
  }


  return Success;
}

void ASTReader::InitializeContext() {  
  // If there's a listener, notify them that we "read" the translation unit.
  if (DeserializationListener)
    DeserializationListener->DeclRead(PREDEF_DECL_TRANSLATION_UNIT_ID, 
                                      Context.getTranslationUnitDecl());

  // Make sure we load the declaration update records for the translation unit,
  // if there are any.
  loadDeclUpdateRecords(PREDEF_DECL_TRANSLATION_UNIT_ID, 
                        Context.getTranslationUnitDecl());
  
  // FIXME: Find a better way to deal with collisions between these
  // built-in types. Right now, we just ignore the problem.
  
  // Load the special types.
  if (SpecialTypes.size() >= NumSpecialTypeIDs) {
    if (Context.getBuiltinVaListType().isNull()) {
      Context.setBuiltinVaListType(
        GetType(SpecialTypes[SPECIAL_TYPE_BUILTIN_VA_LIST]));
    }
    
    if (unsigned String = SpecialTypes[SPECIAL_TYPE_CF_CONSTANT_STRING]) {
      if (!Context.CFConstantStringTypeDecl)
        Context.setCFConstantStringType(GetType(String));
    }
    
    if (unsigned File = SpecialTypes[SPECIAL_TYPE_FILE]) {
      QualType FileType = GetType(File);
      if (FileType.isNull()) {
        Error("FILE type is NULL");
        return;
      }
      
      if (!Context.FILEDecl) {
        if (const TypedefType *Typedef = FileType->getAs<TypedefType>())
          Context.setFILEDecl(Typedef->getDecl());
        else {
          const TagType *Tag = FileType->getAs<TagType>();
          if (!Tag) {
            Error("Invalid FILE type in AST file");
            return;
          }
          Context.setFILEDecl(Tag->getDecl());
        }
      }
    }
    
    if (unsigned Jmp_buf = SpecialTypes[SPECIAL_TYPE_JMP_BUF]) {
      QualType Jmp_bufType = GetType(Jmp_buf);
      if (Jmp_bufType.isNull()) {
        Error("jmp_buf type is NULL");
        return;
      }
      
      if (!Context.jmp_bufDecl) {
        if (const TypedefType *Typedef = Jmp_bufType->getAs<TypedefType>())
          Context.setjmp_bufDecl(Typedef->getDecl());
        else {
          const TagType *Tag = Jmp_bufType->getAs<TagType>();
          if (!Tag) {
            Error("Invalid jmp_buf type in AST file");
            return;
          }
          Context.setjmp_bufDecl(Tag->getDecl());
        }
      }
    }
    
    if (unsigned Sigjmp_buf = SpecialTypes[SPECIAL_TYPE_SIGJMP_BUF]) {
      QualType Sigjmp_bufType = GetType(Sigjmp_buf);
      if (Sigjmp_bufType.isNull()) {
        Error("sigjmp_buf type is NULL");
        return;
      }
      
      if (!Context.sigjmp_bufDecl) {
        if (const TypedefType *Typedef = Sigjmp_bufType->getAs<TypedefType>())
          Context.setsigjmp_bufDecl(Typedef->getDecl());
        else {
          const TagType *Tag = Sigjmp_bufType->getAs<TagType>();
          assert(Tag && "Invalid sigjmp_buf type in AST file");
          Context.setsigjmp_bufDecl(Tag->getDecl());
        }
      }
    }

    if (unsigned ObjCIdRedef
          = SpecialTypes[SPECIAL_TYPE_OBJC_ID_REDEFINITION]) {
      if (Context.ObjCIdRedefinitionType.isNull())
        Context.ObjCIdRedefinitionType = GetType(ObjCIdRedef);
    }

    if (unsigned ObjCClassRedef
          = SpecialTypes[SPECIAL_TYPE_OBJC_CLASS_REDEFINITION]) {
      if (Context.ObjCClassRedefinitionType.isNull())
        Context.ObjCClassRedefinitionType = GetType(ObjCClassRedef);
    }

    if (unsigned ObjCSelRedef
          = SpecialTypes[SPECIAL_TYPE_OBJC_SEL_REDEFINITION]) {
      if (Context.ObjCSelRedefinitionType.isNull())
        Context.ObjCSelRedefinitionType = GetType(ObjCSelRedef);
    }

    if (unsigned Ucontext_t = SpecialTypes[SPECIAL_TYPE_UCONTEXT_T]) {
      QualType Ucontext_tType = GetType(Ucontext_t);
      if (Ucontext_tType.isNull()) {
        Error("ucontext_t type is NULL");
        return;
      }

      if (!Context.ucontext_tDecl) {
        if (const TypedefType *Typedef = Ucontext_tType->getAs<TypedefType>())
          Context.setucontext_tDecl(Typedef->getDecl());
        else {
          const TagType *Tag = Ucontext_tType->getAs<TagType>();
          assert(Tag && "Invalid ucontext_t type in AST file");
          Context.setucontext_tDecl(Tag->getDecl());
        }
      }
    }
  }
  
  ReadPragmaDiagnosticMappings(Context.getDiagnostics());

  // If there were any CUDA special declarations, deserialize them.
  if (!CUDASpecialDeclRefs.empty()) {
    assert(CUDASpecialDeclRefs.size() == 1 && "More decl refs than expected!");
    Context.setcudaConfigureCallDecl(
                           cast<FunctionDecl>(GetDecl(CUDASpecialDeclRefs[0])));
  }
  
  // Re-export any modules that were imported by a non-module AST file.
  for (unsigned I = 0, N = ImportedModules.size(); I != N; ++I) {
    if (Module *Imported = getSubmodule(ImportedModules[I]))
      makeModuleVisible(Imported, Module::AllVisible);
  }
  ImportedModules.clear();
}

void ASTReader::finalizeForWriting() {
  for (HiddenNamesMapType::iterator Hidden = HiddenNamesMap.begin(),
                                 HiddenEnd = HiddenNamesMap.end();
       Hidden != HiddenEnd; ++Hidden) {
    makeNamesVisible(Hidden->second);
  }
  HiddenNamesMap.clear();
}

/// \brief Retrieve the name of the original source file name
/// directly from the AST file, without actually loading the AST
/// file.
std::string ASTReader::getOriginalSourceFile(const std::string &ASTFileName,
                                             FileManager &FileMgr,
                                             DiagnosticsEngine &Diags) {
  // Open the AST file.
  std::string ErrStr;
  OwningPtr<llvm::MemoryBuffer> Buffer;
  Buffer.reset(FileMgr.getBufferForFile(ASTFileName, &ErrStr));
  if (!Buffer) {
    Diags.Report(diag::err_fe_unable_to_read_pch_file) << ErrStr;
    return std::string();
  }

  // Initialize the stream
  llvm::BitstreamReader StreamFile;
  llvm::BitstreamCursor Stream;
  StreamFile.init((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd());
  Stream.init(StreamFile);

  // Sniff for the signature.
  if (Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'P' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(8) != 'H') {
    Diags.Report(diag::err_fe_not_a_pch_file) << ASTFileName;
    return std::string();
  }

  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      unsigned BlockID = Stream.ReadSubBlockID();

      // We only know the AST subblock ID.
      switch (BlockID) {
      case AST_BLOCK_ID:
        if (Stream.EnterSubBlock(AST_BLOCK_ID)) {
          Diags.Report(diag::err_fe_pch_malformed_block) << ASTFileName;
          return std::string();
        }
        break;

      default:
        if (Stream.SkipBlock()) {
          Diags.Report(diag::err_fe_pch_malformed_block) << ASTFileName;
          return std::string();
        }
        break;
      }
      continue;
    }

    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Diags.Report(diag::err_fe_pch_error_at_end_block) << ASTFileName;
        return std::string();
      }
      continue;
    }

    if (Code == llvm::bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    Record.clear();
    const char *BlobStart = 0;
    unsigned BlobLen = 0;
    if (Stream.ReadRecord(Code, Record, &BlobStart, &BlobLen)
          == ORIGINAL_FILE_NAME)
      return std::string(BlobStart, BlobLen);
  }

  return std::string();
}

ASTReader::ASTReadResult ASTReader::ReadSubmoduleBlock(ModuleFile &F) {
  // Enter the submodule block.
  if (F.Stream.EnterSubBlock(SUBMODULE_BLOCK_ID)) {
    Error("malformed submodule block record in AST file");
    return Failure;
  }

  ModuleMap &ModMap = PP.getHeaderSearchInfo().getModuleMap();
  bool First = true;
  Module *CurrentModule = 0;
  RecordData Record;
  while (true) {
    unsigned Code = F.Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (F.Stream.ReadBlockEnd()) {
        Error("error at end of submodule block in AST file");
        return Failure;
      }
      return Success;
    }
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      F.Stream.ReadSubBlockID();
      if (F.Stream.SkipBlock()) {
        Error("malformed block record in AST file");
        return Failure;
      }
      continue;
    }
    
    if (Code == llvm::bitc::DEFINE_ABBREV) {
      F.Stream.ReadAbbrevRecord();
      continue;
    }
    
    // Read a record.
    const char *BlobStart;
    unsigned BlobLen;
    Record.clear();
    switch (F.Stream.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;
      
    case SUBMODULE_DEFINITION: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }

      if (Record.size() < 7) {
        Error("malformed module definition");
        return Failure;
      }
      
      StringRef Name(BlobStart, BlobLen);
      SubmoduleID GlobalID = getGlobalSubmoduleID(F, Record[0]);
      SubmoduleID Parent = getGlobalSubmoduleID(F, Record[1]);
      bool IsFramework = Record[2];
      bool IsExplicit = Record[3];
      bool IsSystem = Record[4];
      bool InferSubmodules = Record[5];
      bool InferExplicitSubmodules = Record[6];
      bool InferExportWildcard = Record[7];
      
      Module *ParentModule = 0;
      if (Parent)
        ParentModule = getSubmodule(Parent);
      
      // Retrieve this (sub)module from the module map, creating it if
      // necessary.
      CurrentModule = ModMap.findOrCreateModule(Name, ParentModule, 
                                                IsFramework, 
                                                IsExplicit).first;
      SubmoduleID GlobalIndex = GlobalID - NUM_PREDEF_SUBMODULE_IDS;
      if (GlobalIndex >= SubmodulesLoaded.size() ||
          SubmodulesLoaded[GlobalIndex]) {
        Error("too many submodules");
        return Failure;
      }
      
      CurrentModule->IsFromModuleFile = true;
      CurrentModule->IsSystem = IsSystem || CurrentModule->IsSystem;
      CurrentModule->InferSubmodules = InferSubmodules;
      CurrentModule->InferExplicitSubmodules = InferExplicitSubmodules;
      CurrentModule->InferExportWildcard = InferExportWildcard;
      if (DeserializationListener)
        DeserializationListener->ModuleRead(GlobalID, CurrentModule);
      
      SubmodulesLoaded[GlobalIndex] = CurrentModule;
      break;
    }
        
    case SUBMODULE_UMBRELLA_HEADER: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }

      if (!CurrentModule)
        break;
      
      StringRef FileName(BlobStart, BlobLen);
      if (const FileEntry *Umbrella = PP.getFileManager().getFile(FileName)) {
        if (!CurrentModule->getUmbrellaHeader())
          ModMap.setUmbrellaHeader(CurrentModule, Umbrella);
        else if (CurrentModule->getUmbrellaHeader() != Umbrella) {
          Error("mismatched umbrella headers in submodule");
          return Failure;
        }
      }
      break;
    }
        
    case SUBMODULE_HEADER: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }

      if (!CurrentModule)
        break;
      
      // FIXME: Be more lazy about this!
      StringRef FileName(BlobStart, BlobLen);
      if (const FileEntry *File = PP.getFileManager().getFile(FileName)) {
        if (std::find(CurrentModule->Headers.begin(), 
                      CurrentModule->Headers.end(), 
                      File) == CurrentModule->Headers.end())
          ModMap.addHeader(CurrentModule, File);
      }
      break;      
    }
        
    case SUBMODULE_UMBRELLA_DIR: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }
      
      if (!CurrentModule)
        break;
      
      StringRef DirName(BlobStart, BlobLen);
      if (const DirectoryEntry *Umbrella
                                  = PP.getFileManager().getDirectory(DirName)) {
        if (!CurrentModule->getUmbrellaDir())
          ModMap.setUmbrellaDir(CurrentModule, Umbrella);
        else if (CurrentModule->getUmbrellaDir() != Umbrella) {
          Error("mismatched umbrella directories in submodule");
          return Failure;
        }
      }
      break;
    }
        
    case SUBMODULE_METADATA: {
      if (!First) {
        Error("submodule metadata record not at beginning of block");
        return Failure;
      }
      First = false;
      
      F.BaseSubmoduleID = getTotalNumSubmodules();
      F.LocalNumSubmodules = Record[0];
      unsigned LocalBaseSubmoduleID = Record[1];
      if (F.LocalNumSubmodules > 0) {
        // Introduce the global -> local mapping for submodules within this 
        // module.
        GlobalSubmoduleMap.insert(std::make_pair(getTotalNumSubmodules()+1,&F));
        
        // Introduce the local -> global mapping for submodules within this 
        // module.
        F.SubmoduleRemap.insertOrReplace(
          std::make_pair(LocalBaseSubmoduleID,
                         F.BaseSubmoduleID - LocalBaseSubmoduleID));
        
        SubmodulesLoaded.resize(SubmodulesLoaded.size() + F.LocalNumSubmodules);
      }      
      break;
    }
        
    case SUBMODULE_IMPORTS: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }
      
      if (!CurrentModule)
        break;
      
      for (unsigned Idx = 0; Idx != Record.size(); ++Idx) {
        UnresolvedModuleImportExport Unresolved;
        Unresolved.File = &F;
        Unresolved.Mod = CurrentModule;
        Unresolved.ID = Record[Idx];
        Unresolved.IsImport = true;
        Unresolved.IsWildcard = false;
        UnresolvedModuleImportExports.push_back(Unresolved);
      }
      break;
    }

    case SUBMODULE_EXPORTS: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }
      
      if (!CurrentModule)
        break;
      
      for (unsigned Idx = 0; Idx + 1 < Record.size(); Idx += 2) {
        UnresolvedModuleImportExport Unresolved;
        Unresolved.File = &F;
        Unresolved.Mod = CurrentModule;
        Unresolved.ID = Record[Idx];
        Unresolved.IsImport = false;
        Unresolved.IsWildcard = Record[Idx + 1];
        UnresolvedModuleImportExports.push_back(Unresolved);
      }
      
      // Once we've loaded the set of exports, there's no reason to keep 
      // the parsed, unresolved exports around.
      CurrentModule->UnresolvedExports.clear();
      break;
    }
    case SUBMODULE_REQUIRES: {
      if (First) {
        Error("missing submodule metadata record at beginning of block");
        return Failure;
      }

      if (!CurrentModule)
        break;

      CurrentModule->addRequirement(StringRef(BlobStart, BlobLen), 
                                    Context.getLangOpts(),
                                    Context.getTargetInfo());
      break;
    }
    }
  }
}

/// \brief Parse the record that corresponds to a LangOptions data
/// structure.
///
/// This routine parses the language options from the AST file and then gives
/// them to the AST listener if one is set.
///
/// \returns true if the listener deems the file unacceptable, false otherwise.
bool ASTReader::ParseLanguageOptions(
                             const SmallVectorImpl<uint64_t> &Record) {
  if (Listener) {
    LangOptions LangOpts;
    unsigned Idx = 0;
#define LANGOPT(Name, Bits, Default, Description) \
  LangOpts.Name = Record[Idx++];
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  LangOpts.set##Name(static_cast<LangOptions::Type>(Record[Idx++]));
#include "clang/Basic/LangOptions.def"
    
    unsigned Length = Record[Idx++];
    LangOpts.CurrentModule.assign(Record.begin() + Idx, 
                                  Record.begin() + Idx + Length);
    return Listener->ReadLanguageOptions(LangOpts);
  }

  return false;
}

std::pair<ModuleFile *, unsigned>
ASTReader::getModulePreprocessedEntity(unsigned GlobalIndex) {
  GlobalPreprocessedEntityMapType::iterator
  I = GlobalPreprocessedEntityMap.find(GlobalIndex);
  assert(I != GlobalPreprocessedEntityMap.end() && 
         "Corrupted global preprocessed entity map");
  ModuleFile *M = I->second;
  unsigned LocalIndex = GlobalIndex - M->BasePreprocessedEntityID;
  return std::make_pair(M, LocalIndex);
}

PreprocessedEntity *ASTReader::ReadPreprocessedEntity(unsigned Index) {
  PreprocessedEntityID PPID = Index+1;
  std::pair<ModuleFile *, unsigned> PPInfo = getModulePreprocessedEntity(Index);
  ModuleFile &M = *PPInfo.first;
  unsigned LocalIndex = PPInfo.second;
  const PPEntityOffset &PPOffs = M.PreprocessedEntityOffsets[LocalIndex];

  SavedStreamPosition SavedPosition(M.PreprocessorDetailCursor);  
  M.PreprocessorDetailCursor.JumpToBit(PPOffs.BitOffset);

  unsigned Code = M.PreprocessorDetailCursor.ReadCode();
  switch (Code) {
  case llvm::bitc::END_BLOCK:
    return 0;
    
  case llvm::bitc::ENTER_SUBBLOCK:
    Error("unexpected subblock record in preprocessor detail block");
    return 0;
      
  case llvm::bitc::DEFINE_ABBREV:
    Error("unexpected abbrevation record in preprocessor detail block");
    return 0;
      
  default:
    break;
  }

  if (!PP.getPreprocessingRecord()) {
    Error("no preprocessing record");
    return 0;
  }
  
  // Read the record.
  SourceRange Range(ReadSourceLocation(M, PPOffs.Begin),
                    ReadSourceLocation(M, PPOffs.End));
  PreprocessingRecord &PPRec = *PP.getPreprocessingRecord();
  const char *BlobStart = 0;
  unsigned BlobLen = 0;
  RecordData Record;
  PreprocessorDetailRecordTypes RecType =
    (PreprocessorDetailRecordTypes)M.PreprocessorDetailCursor.ReadRecord(
                                             Code, Record, BlobStart, BlobLen);
  switch (RecType) {
  case PPD_MACRO_EXPANSION: {
    bool isBuiltin = Record[0];
    IdentifierInfo *Name = 0;
    MacroDefinition *Def = 0;
    if (isBuiltin)
      Name = getLocalIdentifier(M, Record[1]);
    else {
      PreprocessedEntityID
          GlobalID = getGlobalPreprocessedEntityID(M, Record[1]);
      Def =cast<MacroDefinition>(PPRec.getLoadedPreprocessedEntity(GlobalID-1));
    }

    MacroExpansion *ME;
    if (isBuiltin)
      ME = new (PPRec) MacroExpansion(Name, Range);
    else
      ME = new (PPRec) MacroExpansion(Def, Range);

    return ME;
  }
      
  case PPD_MACRO_DEFINITION: {
    // Decode the identifier info and then check again; if the macro is
    // still defined and associated with the identifier,
    IdentifierInfo *II = getLocalIdentifier(M, Record[0]);
    MacroDefinition *MD
      = new (PPRec) MacroDefinition(II, Range);

    if (DeserializationListener)
      DeserializationListener->MacroDefinitionRead(PPID, MD);

    return MD;
  }
      
  case PPD_INCLUSION_DIRECTIVE: {
    const char *FullFileNameStart = BlobStart + Record[0];
    StringRef FullFileName(FullFileNameStart, BlobLen - Record[0]);
    const FileEntry *File = 0;
    if (!FullFileName.empty())
      File = PP.getFileManager().getFile(FullFileName);
    
    // FIXME: Stable encoding
    InclusionDirective::InclusionKind Kind
      = static_cast<InclusionDirective::InclusionKind>(Record[2]);
    InclusionDirective *ID
      = new (PPRec) InclusionDirective(PPRec, Kind,
                                       StringRef(BlobStart, Record[0]),
                                       Record[1],
                                       File,
                                       Range);
    return ID;
  }
  }

  llvm_unreachable("Invalid PreprocessorDetailRecordTypes");
}

/// \brief \arg SLocMapI points at a chunk of a module that contains no
/// preprocessed entities or the entities it contains are not the ones we are
/// looking for. Find the next module that contains entities and return the ID
/// of the first entry.
PreprocessedEntityID ASTReader::findNextPreprocessedEntity(
                       GlobalSLocOffsetMapType::const_iterator SLocMapI) const {
  ++SLocMapI;
  for (GlobalSLocOffsetMapType::const_iterator
         EndI = GlobalSLocOffsetMap.end(); SLocMapI != EndI; ++SLocMapI) {
    ModuleFile &M = *SLocMapI->second;
    if (M.NumPreprocessedEntities)
      return getGlobalPreprocessedEntityID(M, M.BasePreprocessedEntityID);
  }

  return getTotalNumPreprocessedEntities();
}

namespace {

template <unsigned PPEntityOffset::*PPLoc>
struct PPEntityComp {
  const ASTReader &Reader;
  ModuleFile &M;

  PPEntityComp(const ASTReader &Reader, ModuleFile &M) : Reader(Reader), M(M) { }

  bool operator()(const PPEntityOffset &L, const PPEntityOffset &R) const {
    SourceLocation LHS = getLoc(L);
    SourceLocation RHS = getLoc(R);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  bool operator()(const PPEntityOffset &L, SourceLocation RHS) const {
    SourceLocation LHS = getLoc(L);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  bool operator()(SourceLocation LHS, const PPEntityOffset &R) const {
    SourceLocation RHS = getLoc(R);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  SourceLocation getLoc(const PPEntityOffset &PPE) const {
    return Reader.ReadSourceLocation(M, PPE.*PPLoc);
  }
};

}

/// \brief Returns the first preprocessed entity ID that ends after \arg BLoc.
PreprocessedEntityID
ASTReader::findBeginPreprocessedEntity(SourceLocation BLoc) const {
  if (SourceMgr.isLocalSourceLocation(BLoc))
    return getTotalNumPreprocessedEntities();

  GlobalSLocOffsetMapType::const_iterator
    SLocMapI = GlobalSLocOffsetMap.find(SourceManager::MaxLoadedOffset -
                                        BLoc.getOffset());
  assert(SLocMapI != GlobalSLocOffsetMap.end() &&
         "Corrupted global sloc offset map");

  if (SLocMapI->second->NumPreprocessedEntities == 0)
    return findNextPreprocessedEntity(SLocMapI);

  ModuleFile &M = *SLocMapI->second;
  typedef const PPEntityOffset *pp_iterator;
  pp_iterator pp_begin = M.PreprocessedEntityOffsets;
  pp_iterator pp_end = pp_begin + M.NumPreprocessedEntities;

  size_t Count = M.NumPreprocessedEntities;
  size_t Half;
  pp_iterator First = pp_begin;
  pp_iterator PPI;

  // Do a binary search manually instead of using std::lower_bound because
  // The end locations of entities may be unordered (when a macro expansion
  // is inside another macro argument), but for this case it is not important
  // whether we get the first macro expansion or its containing macro.
  while (Count > 0) {
    Half = Count/2;
    PPI = First;
    std::advance(PPI, Half);
    if (SourceMgr.isBeforeInTranslationUnit(ReadSourceLocation(M, PPI->End),
                                            BLoc)){
      First = PPI;
      ++First;
      Count = Count - Half - 1;
    } else
      Count = Half;
  }

  if (PPI == pp_end)
    return findNextPreprocessedEntity(SLocMapI);

  return getGlobalPreprocessedEntityID(M,
                                 M.BasePreprocessedEntityID + (PPI - pp_begin));
}

/// \brief Returns the first preprocessed entity ID that begins after \arg ELoc.
PreprocessedEntityID
ASTReader::findEndPreprocessedEntity(SourceLocation ELoc) const {
  if (SourceMgr.isLocalSourceLocation(ELoc))
    return getTotalNumPreprocessedEntities();

  GlobalSLocOffsetMapType::const_iterator
    SLocMapI = GlobalSLocOffsetMap.find(SourceManager::MaxLoadedOffset -
                                        ELoc.getOffset());
  assert(SLocMapI != GlobalSLocOffsetMap.end() &&
         "Corrupted global sloc offset map");

  if (SLocMapI->second->NumPreprocessedEntities == 0)
    return findNextPreprocessedEntity(SLocMapI);

  ModuleFile &M = *SLocMapI->second;
  typedef const PPEntityOffset *pp_iterator;
  pp_iterator pp_begin = M.PreprocessedEntityOffsets;
  pp_iterator pp_end = pp_begin + M.NumPreprocessedEntities;
  pp_iterator PPI =
      std::upper_bound(pp_begin, pp_end, ELoc,
                       PPEntityComp<&PPEntityOffset::Begin>(*this, M));

  if (PPI == pp_end)
    return findNextPreprocessedEntity(SLocMapI);

  return getGlobalPreprocessedEntityID(M,
                                 M.BasePreprocessedEntityID + (PPI - pp_begin));
}

/// \brief Returns a pair of [Begin, End) indices of preallocated
/// preprocessed entities that \arg Range encompasses.
std::pair<unsigned, unsigned>
    ASTReader::findPreprocessedEntitiesInRange(SourceRange Range) {
  if (Range.isInvalid())
    return std::make_pair(0,0);
  assert(!SourceMgr.isBeforeInTranslationUnit(Range.getEnd(),Range.getBegin()));

  PreprocessedEntityID BeginID = findBeginPreprocessedEntity(Range.getBegin());
  PreprocessedEntityID EndID = findEndPreprocessedEntity(Range.getEnd());
  return std::make_pair(BeginID, EndID);
}

/// \brief Optionally returns true or false if the preallocated preprocessed
/// entity with index \arg Index came from file \arg FID.
llvm::Optional<bool> ASTReader::isPreprocessedEntityInFileID(unsigned Index,
                                                             FileID FID) {
  if (FID.isInvalid())
    return false;

  std::pair<ModuleFile *, unsigned> PPInfo = getModulePreprocessedEntity(Index);
  ModuleFile &M = *PPInfo.first;
  unsigned LocalIndex = PPInfo.second;
  const PPEntityOffset &PPOffs = M.PreprocessedEntityOffsets[LocalIndex];
  
  SourceLocation Loc = ReadSourceLocation(M, PPOffs.Begin);
  if (Loc.isInvalid())
    return false;
  
  if (SourceMgr.isInFileID(SourceMgr.getFileLoc(Loc), FID))
    return true;
  else
    return false;
}

namespace {
  /// \brief Visitor used to search for information about a header file.
  class HeaderFileInfoVisitor {
    ASTReader &Reader;
    const FileEntry *FE;
    
    llvm::Optional<HeaderFileInfo> HFI;
    
  public:
    HeaderFileInfoVisitor(ASTReader &Reader, const FileEntry *FE)
      : Reader(Reader), FE(FE) { }
    
    static bool visit(ModuleFile &M, void *UserData) {
      HeaderFileInfoVisitor *This
        = static_cast<HeaderFileInfoVisitor *>(UserData);
      
      HeaderFileInfoTrait Trait(This->Reader, M, 
                                &This->Reader.getPreprocessor().getHeaderSearchInfo(),
                                M.HeaderFileFrameworkStrings,
                                This->FE->getName());
      
      HeaderFileInfoLookupTable *Table
        = static_cast<HeaderFileInfoLookupTable *>(M.HeaderFileInfoTable);
      if (!Table)
        return false;

      // Look in the on-disk hash table for an entry for this file name.
      HeaderFileInfoLookupTable::iterator Pos = Table->find(This->FE->getName(),
                                                            &Trait);
      if (Pos == Table->end())
        return false;

      This->HFI = *Pos;
      return true;
    }
    
    llvm::Optional<HeaderFileInfo> getHeaderFileInfo() const { return HFI; }
  };
}

HeaderFileInfo ASTReader::GetHeaderFileInfo(const FileEntry *FE) {
  HeaderFileInfoVisitor Visitor(*this, FE);
  ModuleMgr.visit(&HeaderFileInfoVisitor::visit, &Visitor);
  if (llvm::Optional<HeaderFileInfo> HFI = Visitor.getHeaderFileInfo()) {
    if (Listener)
      Listener->ReadHeaderFileInfo(*HFI, FE->getUID());
    return *HFI;
  }
  
  return HeaderFileInfo();
}

void ASTReader::ReadPragmaDiagnosticMappings(DiagnosticsEngine &Diag) {
  for (ModuleIterator I = ModuleMgr.begin(), E = ModuleMgr.end(); I != E; ++I) {
    ModuleFile &F = *(*I);
    unsigned Idx = 0;
    while (Idx < F.PragmaDiagMappings.size()) {
      SourceLocation Loc = ReadSourceLocation(F, F.PragmaDiagMappings[Idx++]);
      Diag.DiagStates.push_back(*Diag.GetCurDiagState());
      Diag.DiagStatePoints.push_back(
          DiagnosticsEngine::DiagStatePoint(&Diag.DiagStates.back(),
                                            FullSourceLoc(Loc, SourceMgr)));
      while (1) {
        assert(Idx < F.PragmaDiagMappings.size() &&
               "Invalid data, didn't find '-1' marking end of diag/map pairs");
        if (Idx >= F.PragmaDiagMappings.size()) {
          break; // Something is messed up but at least avoid infinite loop in
                 // release build.
        }
        unsigned DiagID = F.PragmaDiagMappings[Idx++];
        if (DiagID == (unsigned)-1) {
          break; // no more diag/map pairs for this location.
        }
        diag::Mapping Map = (diag::Mapping)F.PragmaDiagMappings[Idx++];
        DiagnosticMappingInfo MappingInfo = Diag.makeMappingInfo(Map, Loc);
        Diag.GetCurDiagState()->setMappingInfo(DiagID, MappingInfo);
      }
    }
  }
}

/// \brief Get the correct cursor and offset for loading a type.
ASTReader::RecordLocation ASTReader::TypeCursorForIndex(unsigned Index) {
  GlobalTypeMapType::iterator I = GlobalTypeMap.find(Index);
  assert(I != GlobalTypeMap.end() && "Corrupted global type map");
  ModuleFile *M = I->second;
  return RecordLocation(M, M->TypeOffsets[Index - M->BaseTypeIndex]);
}

/// \brief Read and return the type with the given index..
///
/// The index is the type ID, shifted and minus the number of predefs. This
/// routine actually reads the record corresponding to the type at the given
/// location. It is a helper routine for GetType, which deals with reading type
/// IDs.
QualType ASTReader::readTypeRecord(unsigned Index) {
  RecordLocation Loc = TypeCursorForIndex(Index);
  llvm::BitstreamCursor &DeclsCursor = Loc.F->DeclsCursor;

  // Keep track of where we are in the stream, then jump back there
  // after reading this type.
  SavedStreamPosition SavedPosition(DeclsCursor);

  ReadingKindTracker ReadingKind(Read_Type, *this);

  // Note that we are loading a type record.
  Deserializing AType(this);

  unsigned Idx = 0;
  DeclsCursor.JumpToBit(Loc.Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  switch ((TypeCode)DeclsCursor.ReadRecord(Code, Record)) {
  case TYPE_EXT_QUAL: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of extended qualifier type");
      return QualType();
    }
    QualType Base = readType(*Loc.F, Record, Idx);
    Qualifiers Quals = Qualifiers::fromOpaqueValue(Record[Idx++]);
    return Context.getQualifiedType(Base, Quals);
  }

  case TYPE_COMPLEX: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of complex type");
      return QualType();
    }
    QualType ElemType = readType(*Loc.F, Record, Idx);
    return Context.getComplexType(ElemType);
  }

  case TYPE_POINTER: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context.getPointerType(PointeeType);
  }

  case TYPE_BLOCK_POINTER: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of block pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context.getBlockPointerType(PointeeType);
  }

  case TYPE_LVALUE_REFERENCE: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of lvalue reference type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context.getLValueReferenceType(PointeeType, Record[1]);
  }

  case TYPE_RVALUE_REFERENCE: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of rvalue reference type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    return Context.getRValueReferenceType(PointeeType);
  }

  case TYPE_MEMBER_POINTER: {
    if (Record.size() != 2) {
      Error("Incorrect encoding of member pointer type");
      return QualType();
    }
    QualType PointeeType = readType(*Loc.F, Record, Idx);
    QualType ClassType = readType(*Loc.F, Record, Idx);
    if (PointeeType.isNull() || ClassType.isNull())
      return QualType();
    
    return Context.getMemberPointerType(PointeeType, ClassType.getTypePtr());
  }

  case TYPE_CONSTANT_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    unsigned Idx = 3;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context.getConstantArrayType(ElementType, Size,
                                         ASM, IndexTypeQuals);
  }

  case TYPE_INCOMPLETE_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    return Context.getIncompleteArrayType(ElementType, ASM, IndexTypeQuals);
  }

  case TYPE_VARIABLE_ARRAY: {
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    SourceLocation LBLoc = ReadSourceLocation(*Loc.F, Record[3]);
    SourceLocation RBLoc = ReadSourceLocation(*Loc.F, Record[4]);
    return Context.getVariableArrayType(ElementType, ReadExpr(*Loc.F),
                                         ASM, IndexTypeQuals,
                                         SourceRange(LBLoc, RBLoc));
  }

  case TYPE_VECTOR: {
    if (Record.size() != 3) {
      Error("incorrect encoding of vector type in AST file");
      return QualType();
    }

    QualType ElementType = readType(*Loc.F, Record, Idx);
    unsigned NumElements = Record[1];
    unsigned VecKind = Record[2];
    return Context.getVectorType(ElementType, NumElements,
                                  (VectorType::VectorKind)VecKind);
  }

  case TYPE_EXT_VECTOR: {
    if (Record.size() != 3) {
      Error("incorrect encoding of extended vector type in AST file");
      return QualType();
    }

    QualType ElementType = readType(*Loc.F, Record, Idx);
    unsigned NumElements = Record[1];
    return Context.getExtVectorType(ElementType, NumElements);
  }

  case TYPE_FUNCTION_NO_PROTO: {
    if (Record.size() != 6) {
      Error("incorrect encoding of no-proto function type");
      return QualType();
    }
    QualType ResultType = readType(*Loc.F, Record, Idx);
    FunctionType::ExtInfo Info(Record[1], Record[2], Record[3],
                               (CallingConv)Record[4], Record[5]);
    return Context.getFunctionNoProtoType(ResultType, Info);
  }

  case TYPE_FUNCTION_PROTO: {
    QualType ResultType = readType(*Loc.F, Record, Idx);

    FunctionProtoType::ExtProtoInfo EPI;
    EPI.ExtInfo = FunctionType::ExtInfo(/*noreturn*/ Record[1],
                                        /*hasregparm*/ Record[2],
                                        /*regparm*/ Record[3],
                                        static_cast<CallingConv>(Record[4]),
                                        /*produces*/ Record[5]);

    unsigned Idx = 6;
    unsigned NumParams = Record[Idx++];
    SmallVector<QualType, 16> ParamTypes;
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(readType(*Loc.F, Record, Idx));

    EPI.Variadic = Record[Idx++];
    EPI.HasTrailingReturn = Record[Idx++];
    EPI.TypeQuals = Record[Idx++];
    EPI.RefQualifier = static_cast<RefQualifierKind>(Record[Idx++]);
    ExceptionSpecificationType EST =
        static_cast<ExceptionSpecificationType>(Record[Idx++]);
    EPI.ExceptionSpecType = EST;
    SmallVector<QualType, 2> Exceptions;
    if (EST == EST_Dynamic) {
      EPI.NumExceptions = Record[Idx++];
      for (unsigned I = 0; I != EPI.NumExceptions; ++I)
        Exceptions.push_back(readType(*Loc.F, Record, Idx));
      EPI.Exceptions = Exceptions.data();
    } else if (EST == EST_ComputedNoexcept) {
      EPI.NoexceptExpr = ReadExpr(*Loc.F);
    } else if (EST == EST_Uninstantiated) {
      EPI.ExceptionSpecDecl = ReadDeclAs<FunctionDecl>(*Loc.F, Record, Idx);
      EPI.ExceptionSpecTemplate = ReadDeclAs<FunctionDecl>(*Loc.F, Record, Idx);
    }
    return Context.getFunctionType(ResultType, ParamTypes.data(), NumParams,
                                    EPI);
  }

  case TYPE_UNRESOLVED_USING: {
    unsigned Idx = 0;
    return Context.getTypeDeclType(
                  ReadDeclAs<UnresolvedUsingTypenameDecl>(*Loc.F, Record, Idx));
  }
      
  case TYPE_TYPEDEF: {
    if (Record.size() != 2) {
      Error("incorrect encoding of typedef type");
      return QualType();
    }
    unsigned Idx = 0;
    TypedefNameDecl *Decl = ReadDeclAs<TypedefNameDecl>(*Loc.F, Record, Idx);
    QualType Canonical = readType(*Loc.F, Record, Idx);
    if (!Canonical.isNull())
      Canonical = Context.getCanonicalType(Canonical);
    return Context.getTypedefType(Decl, Canonical);
  }

  case TYPE_TYPEOF_EXPR:
    return Context.getTypeOfExprType(ReadExpr(*Loc.F));

  case TYPE_TYPEOF: {
    if (Record.size() != 1) {
      Error("incorrect encoding of typeof(type) in AST file");
      return QualType();
    }
    QualType UnderlyingType = readType(*Loc.F, Record, Idx);
    return Context.getTypeOfType(UnderlyingType);
  }

  case TYPE_DECLTYPE: {
    QualType UnderlyingType = readType(*Loc.F, Record, Idx);
    return Context.getDecltypeType(ReadExpr(*Loc.F), UnderlyingType);
  }

  case TYPE_UNARY_TRANSFORM: {
    QualType BaseType = readType(*Loc.F, Record, Idx);
    QualType UnderlyingType = readType(*Loc.F, Record, Idx);
    UnaryTransformType::UTTKind UKind = (UnaryTransformType::UTTKind)Record[2];
    return Context.getUnaryTransformType(BaseType, UnderlyingType, UKind);
  }

  case TYPE_AUTO:
    return Context.getAutoType(readType(*Loc.F, Record, Idx));

  case TYPE_RECORD: {
    if (Record.size() != 2) {
      Error("incorrect encoding of record type");
      return QualType();
    }
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    RecordDecl *RD = ReadDeclAs<RecordDecl>(*Loc.F, Record, Idx);
    RD = cast_or_null<RecordDecl>(RD->getCanonicalDecl());
    QualType T = Context.getRecordType(RD);
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }

  case TYPE_ENUM: {
    if (Record.size() != 2) {
      Error("incorrect encoding of enum type");
      return QualType();
    }
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    QualType T
      = Context.getEnumType(ReadDeclAs<EnumDecl>(*Loc.F, Record, Idx));
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }

  case TYPE_ATTRIBUTED: {
    if (Record.size() != 3) {
      Error("incorrect encoding of attributed type");
      return QualType();
    }
    QualType modifiedType = readType(*Loc.F, Record, Idx);
    QualType equivalentType = readType(*Loc.F, Record, Idx);
    AttributedType::Kind kind = static_cast<AttributedType::Kind>(Record[2]);
    return Context.getAttributedType(kind, modifiedType, equivalentType);
  }

  case TYPE_PAREN: {
    if (Record.size() != 1) {
      Error("incorrect encoding of paren type");
      return QualType();
    }
    QualType InnerType = readType(*Loc.F, Record, Idx);
    return Context.getParenType(InnerType);
  }

  case TYPE_PACK_EXPANSION: {
    if (Record.size() != 2) {
      Error("incorrect encoding of pack expansion type");
      return QualType();
    }
    QualType Pattern = readType(*Loc.F, Record, Idx);
    if (Pattern.isNull())
      return QualType();
    llvm::Optional<unsigned> NumExpansions;
    if (Record[1])
      NumExpansions = Record[1] - 1;
    return Context.getPackExpansionType(Pattern, NumExpansions);
  }

  case TYPE_ELABORATED: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    QualType NamedType = readType(*Loc.F, Record, Idx);
    return Context.getElaboratedType(Keyword, NNS, NamedType);
  }

  case TYPE_OBJC_INTERFACE: {
    unsigned Idx = 0;
    ObjCInterfaceDecl *ItfD
      = ReadDeclAs<ObjCInterfaceDecl>(*Loc.F, Record, Idx);
    return Context.getObjCInterfaceType(ItfD->getCanonicalDecl());
  }

  case TYPE_OBJC_OBJECT: {
    unsigned Idx = 0;
    QualType Base = readType(*Loc.F, Record, Idx);
    unsigned NumProtos = Record[Idx++];
    SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(ReadDeclAs<ObjCProtocolDecl>(*Loc.F, Record, Idx));
    return Context.getObjCObjectType(Base, Protos.data(), NumProtos);
  }

  case TYPE_OBJC_OBJECT_POINTER: {
    unsigned Idx = 0;
    QualType Pointee = readType(*Loc.F, Record, Idx);
    return Context.getObjCObjectPointerType(Pointee);
  }

  case TYPE_SUBST_TEMPLATE_TYPE_PARM: {
    unsigned Idx = 0;
    QualType Parm = readType(*Loc.F, Record, Idx);
    QualType Replacement = readType(*Loc.F, Record, Idx);
    return
      Context.getSubstTemplateTypeParmType(cast<TemplateTypeParmType>(Parm),
                                            Replacement);
  }

  case TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK: {
    unsigned Idx = 0;
    QualType Parm = readType(*Loc.F, Record, Idx);
    TemplateArgument ArgPack = ReadTemplateArgument(*Loc.F, Record, Idx);
    return Context.getSubstTemplateTypeParmPackType(
                                               cast<TemplateTypeParmType>(Parm),
                                                     ArgPack);
  }

  case TYPE_INJECTED_CLASS_NAME: {
    CXXRecordDecl *D = ReadDeclAs<CXXRecordDecl>(*Loc.F, Record, Idx);
    QualType TST = readType(*Loc.F, Record, Idx); // probably derivable
    // FIXME: ASTContext::getInjectedClassNameType is not currently suitable
    // for AST reading, too much interdependencies.
    return
      QualType(new (Context, TypeAlignment) InjectedClassNameType(D, TST), 0);
  }

  case TYPE_TEMPLATE_TYPE_PARM: {
    unsigned Idx = 0;
    unsigned Depth = Record[Idx++];
    unsigned Index = Record[Idx++];
    bool Pack = Record[Idx++];
    TemplateTypeParmDecl *D
      = ReadDeclAs<TemplateTypeParmDecl>(*Loc.F, Record, Idx);
    return Context.getTemplateTypeParmType(Depth, Index, Pack, D);
  }

  case TYPE_DEPENDENT_NAME: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    const IdentifierInfo *Name = this->GetIdentifierInfo(*Loc.F, Record, Idx);
    QualType Canon = readType(*Loc.F, Record, Idx);
    if (!Canon.isNull())
      Canon = Context.getCanonicalType(Canon);
    return Context.getDependentNameType(Keyword, NNS, Name, Canon);
  }

  case TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION: {
    unsigned Idx = 0;
    ElaboratedTypeKeyword Keyword = (ElaboratedTypeKeyword)Record[Idx++];
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(*Loc.F, Record, Idx);
    const IdentifierInfo *Name = this->GetIdentifierInfo(*Loc.F, Record, Idx);
    unsigned NumArgs = Record[Idx++];
    SmallVector<TemplateArgument, 8> Args;
    Args.reserve(NumArgs);
    while (NumArgs--)
      Args.push_back(ReadTemplateArgument(*Loc.F, Record, Idx));
    return Context.getDependentTemplateSpecializationType(Keyword, NNS, Name,
                                                      Args.size(), Args.data());
  }

  case TYPE_DEPENDENT_SIZED_ARRAY: {
    unsigned Idx = 0;

    // ArrayType
    QualType ElementType = readType(*Loc.F, Record, Idx);
    ArrayType::ArraySizeModifier ASM
      = (ArrayType::ArraySizeModifier)Record[Idx++];
    unsigned IndexTypeQuals = Record[Idx++];

    // DependentSizedArrayType
    Expr *NumElts = ReadExpr(*Loc.F);
    SourceRange Brackets = ReadSourceRange(*Loc.F, Record, Idx);

    return Context.getDependentSizedArrayType(ElementType, NumElts, ASM,
                                               IndexTypeQuals, Brackets);
  }

  case TYPE_TEMPLATE_SPECIALIZATION: {
    unsigned Idx = 0;
    bool IsDependent = Record[Idx++];
    TemplateName Name = ReadTemplateName(*Loc.F, Record, Idx);
    SmallVector<TemplateArgument, 8> Args;
    ReadTemplateArgumentList(Args, *Loc.F, Record, Idx);
    QualType Underlying = readType(*Loc.F, Record, Idx);
    QualType T;
    if (Underlying.isNull())
      T = Context.getCanonicalTemplateSpecializationType(Name, Args.data(),
                                                          Args.size());
    else
      T = Context.getTemplateSpecializationType(Name, Args.data(),
                                                 Args.size(), Underlying);
    const_cast<Type*>(T.getTypePtr())->setDependent(IsDependent);
    return T;
  }

  case TYPE_ATOMIC: {
    if (Record.size() != 1) {
      Error("Incorrect encoding of atomic type");
      return QualType();
    }
    QualType ValueType = readType(*Loc.F, Record, Idx);
    return Context.getAtomicType(ValueType);
  }
  }
  llvm_unreachable("Invalid TypeCode!");
}

class clang::TypeLocReader : public TypeLocVisitor<TypeLocReader> {
  ASTReader &Reader;
  ModuleFile &F;
  llvm::BitstreamCursor &DeclsCursor;
  const ASTReader::RecordData &Record;
  unsigned &Idx;

  SourceLocation ReadSourceLocation(const ASTReader::RecordData &R,
                                    unsigned &I) {
    return Reader.ReadSourceLocation(F, R, I);
  }

  template<typename T>
  T *ReadDeclAs(const ASTReader::RecordData &Record, unsigned &Idx) {
    return Reader.ReadDeclAs<T>(F, Record, Idx);
  }
  
public:
  TypeLocReader(ASTReader &Reader, ModuleFile &F,
                const ASTReader::RecordData &Record, unsigned &Idx)
    : Reader(Reader), F(F), DeclsCursor(F.DeclsCursor), Record(Record), Idx(Idx)
  { }

  // We want compile-time assurance that we've enumerated all of
  // these, so unfortunately we have to declare them first, then
  // define them out-of-line.
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
  void Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc);
#include "clang/AST/TypeLocNodes.def"

  void VisitFunctionTypeLoc(FunctionTypeLoc);
  void VisitArrayTypeLoc(ArrayTypeLoc);
};

void TypeLocReader::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  // nothing to do
}
void TypeLocReader::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  TL.setBuiltinLoc(ReadSourceLocation(Record, Idx));
  if (TL.needsExtraLocalData()) {
    TL.setWrittenTypeSpec(static_cast<DeclSpec::TST>(Record[Idx++]));
    TL.setWrittenSignSpec(static_cast<DeclSpec::TSS>(Record[Idx++]));
    TL.setWrittenWidthSpec(static_cast<DeclSpec::TSW>(Record[Idx++]));
    TL.setModeAttr(Record[Idx++]);
  }
}
void TypeLocReader::VisitComplexTypeLoc(ComplexTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitPointerTypeLoc(PointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  TL.setCaretLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  TL.setAmpLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  TL.setAmpAmpLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
  TL.setClassTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  TL.setLBracketLoc(ReadSourceLocation(Record, Idx));
  TL.setRBracketLoc(ReadSourceLocation(Record, Idx));
  if (Record[Idx++])
    TL.setSizeExpr(Reader.ReadExpr(F));
  else
    TL.setSizeExpr(0);
}
void TypeLocReader::VisitConstantArrayTypeLoc(ConstantArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitIncompleteArrayTypeLoc(IncompleteArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitVariableArrayTypeLoc(VariableArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitDependentSizedArrayTypeLoc(
                                            DependentSizedArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocReader::VisitDependentSizedExtVectorTypeLoc(
                                        DependentSizedExtVectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitVectorTypeLoc(VectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitExtVectorTypeLoc(ExtVectorTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitFunctionTypeLoc(FunctionTypeLoc TL) {
  TL.setLocalRangeBegin(ReadSourceLocation(Record, Idx));
  TL.setLocalRangeEnd(ReadSourceLocation(Record, Idx));
  TL.setTrailingReturn(Record[Idx++]);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i) {
    TL.setArg(i, ReadDeclAs<ParmVarDecl>(Record, Idx));
  }
}
void TypeLocReader::VisitFunctionProtoTypeLoc(FunctionProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocReader::VisitFunctionNoProtoTypeLoc(FunctionNoProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocReader::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  TL.setTypeofLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  TL.setTypeofLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
  TL.setUnderlyingTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitUnaryTransformTypeLoc(UnaryTransformTypeLoc TL) {
  TL.setKWLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
  TL.setUnderlyingTInfo(Reader.GetTypeSourceInfo(F, Record, Idx));
}
void TypeLocReader::VisitAutoTypeLoc(AutoTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitRecordTypeLoc(RecordTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitEnumTypeLoc(EnumTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitAttributedTypeLoc(AttributedTypeLoc TL) {
  TL.setAttrNameLoc(ReadSourceLocation(Record, Idx));
  if (TL.hasAttrOperand()) {
    SourceRange range;
    range.setBegin(ReadSourceLocation(Record, Idx));
    range.setEnd(ReadSourceLocation(Record, Idx));
    TL.setAttrOperandParensRange(range);
  }
  if (TL.hasAttrExprOperand()) {
    if (Record[Idx++])
      TL.setAttrExprOperand(Reader.ReadExpr(F));
    else
      TL.setAttrExprOperand(0);
  } else if (TL.hasAttrEnumOperand())
    TL.setAttrEnumOperandLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitSubstTemplateTypeParmTypeLoc(
                                            SubstTemplateTypeParmTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitSubstTemplateTypeParmPackTypeLoc(
                                          SubstTemplateTypeParmPackTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitTemplateSpecializationTypeLoc(
                                           TemplateSpecializationTypeLoc TL) {
  TL.setTemplateKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setTemplateNameLoc(ReadSourceLocation(Record, Idx));
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    TL.setArgLocInfo(i,
        Reader.GetTemplateArgumentLocInfo(F,
                                          TL.getTypePtr()->getArg(i).getKind(),
                                          Record, Idx));
}
void TypeLocReader::VisitParenTypeLoc(ParenTypeLoc TL) {
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
  TL.setElaboratedKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
}
void TypeLocReader::VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
  TL.setElaboratedKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitDependentTemplateSpecializationTypeLoc(
       DependentTemplateSpecializationTypeLoc TL) {
  TL.setElaboratedKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setQualifierLoc(Reader.ReadNestedNameSpecifierLoc(F, Record, Idx));
  TL.setTemplateKeywordLoc(ReadSourceLocation(Record, Idx));
  TL.setTemplateNameLoc(ReadSourceLocation(Record, Idx));
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I)
    TL.setArgLocInfo(I,
        Reader.GetTemplateArgumentLocInfo(F,
                                          TL.getTypePtr()->getArg(I).getKind(),
                                          Record, Idx));
}
void TypeLocReader::VisitPackExpansionTypeLoc(PackExpansionTypeLoc TL) {
  TL.setEllipsisLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  TL.setNameLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  TL.setHasBaseTypeAsWritten(Record[Idx++]);
  TL.setLAngleLoc(ReadSourceLocation(Record, Idx));
  TL.setRAngleLoc(ReadSourceLocation(Record, Idx));
  for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i)
    TL.setProtocolLoc(i, ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  TL.setStarLoc(ReadSourceLocation(Record, Idx));
}
void TypeLocReader::VisitAtomicTypeLoc(AtomicTypeLoc TL) {
  TL.setKWLoc(ReadSourceLocation(Record, Idx));
  TL.setLParenLoc(ReadSourceLocation(Record, Idx));
  TL.setRParenLoc(ReadSourceLocation(Record, Idx));
}

TypeSourceInfo *ASTReader::GetTypeSourceInfo(ModuleFile &F,
                                             const RecordData &Record,
                                             unsigned &Idx) {
  QualType InfoTy = readType(F, Record, Idx);
  if (InfoTy.isNull())
    return 0;

  TypeSourceInfo *TInfo = getContext().CreateTypeSourceInfo(InfoTy);
  TypeLocReader TLR(*this, F, Record, Idx);
  for (TypeLoc TL = TInfo->getTypeLoc(); !TL.isNull(); TL = TL.getNextTypeLoc())
    TLR.Visit(TL);
  return TInfo;
}

QualType ASTReader::GetType(TypeID ID) {
  unsigned FastQuals = ID & Qualifiers::FastMask;
  unsigned Index = ID >> Qualifiers::FastWidth;

  if (Index < NUM_PREDEF_TYPE_IDS) {
    QualType T;
    switch ((PredefinedTypeIDs)Index) {
    case PREDEF_TYPE_NULL_ID: return QualType();
    case PREDEF_TYPE_VOID_ID: T = Context.VoidTy; break;
    case PREDEF_TYPE_BOOL_ID: T = Context.BoolTy; break;

    case PREDEF_TYPE_CHAR_U_ID:
    case PREDEF_TYPE_CHAR_S_ID:
      // FIXME: Check that the signedness of CharTy is correct!
      T = Context.CharTy;
      break;

    case PREDEF_TYPE_UCHAR_ID:      T = Context.UnsignedCharTy;     break;
    case PREDEF_TYPE_USHORT_ID:     T = Context.UnsignedShortTy;    break;
    case PREDEF_TYPE_UINT_ID:       T = Context.UnsignedIntTy;      break;
    case PREDEF_TYPE_ULONG_ID:      T = Context.UnsignedLongTy;     break;
    case PREDEF_TYPE_ULONGLONG_ID:  T = Context.UnsignedLongLongTy; break;
    case PREDEF_TYPE_UINT128_ID:    T = Context.UnsignedInt128Ty;   break;
    case PREDEF_TYPE_SCHAR_ID:      T = Context.SignedCharTy;       break;
    case PREDEF_TYPE_WCHAR_ID:      T = Context.WCharTy;            break;
    case PREDEF_TYPE_SHORT_ID:      T = Context.ShortTy;            break;
    case PREDEF_TYPE_INT_ID:        T = Context.IntTy;              break;
    case PREDEF_TYPE_LONG_ID:       T = Context.LongTy;             break;
    case PREDEF_TYPE_LONGLONG_ID:   T = Context.LongLongTy;         break;
    case PREDEF_TYPE_INT128_ID:     T = Context.Int128Ty;           break;
    case PREDEF_TYPE_HALF_ID:       T = Context.HalfTy;             break;
    case PREDEF_TYPE_FLOAT_ID:      T = Context.FloatTy;            break;
    case PREDEF_TYPE_DOUBLE_ID:     T = Context.DoubleTy;           break;
    case PREDEF_TYPE_LONGDOUBLE_ID: T = Context.LongDoubleTy;       break;
    case PREDEF_TYPE_OVERLOAD_ID:   T = Context.OverloadTy;         break;
    case PREDEF_TYPE_BOUND_MEMBER:  T = Context.BoundMemberTy;      break;
    case PREDEF_TYPE_PSEUDO_OBJECT: T = Context.PseudoObjectTy;     break;
    case PREDEF_TYPE_DEPENDENT_ID:  T = Context.DependentTy;        break;
    case PREDEF_TYPE_UNKNOWN_ANY:   T = Context.UnknownAnyTy;       break;
    case PREDEF_TYPE_NULLPTR_ID:    T = Context.NullPtrTy;          break;
    case PREDEF_TYPE_CHAR16_ID:     T = Context.Char16Ty;           break;
    case PREDEF_TYPE_CHAR32_ID:     T = Context.Char32Ty;           break;
    case PREDEF_TYPE_OBJC_ID:       T = Context.ObjCBuiltinIdTy;    break;
    case PREDEF_TYPE_OBJC_CLASS:    T = Context.ObjCBuiltinClassTy; break;
    case PREDEF_TYPE_OBJC_SEL:      T = Context.ObjCBuiltinSelTy;   break;
    case PREDEF_TYPE_AUTO_DEDUCT:   T = Context.getAutoDeductType(); break;
        
    case PREDEF_TYPE_AUTO_RREF_DEDUCT: 
      T = Context.getAutoRRefDeductType(); 
      break;

    case PREDEF_TYPE_ARC_UNBRIDGED_CAST:
      T = Context.ARCUnbridgedCastTy;
      break;

    }

    assert(!T.isNull() && "Unknown predefined type");
    return T.withFastQualifiers(FastQuals);
  }

  Index -= NUM_PREDEF_TYPE_IDS;
  assert(Index < TypesLoaded.size() && "Type index out-of-range");
  if (TypesLoaded[Index].isNull()) {
    TypesLoaded[Index] = readTypeRecord(Index);
    if (TypesLoaded[Index].isNull())
      return QualType();

    TypesLoaded[Index]->setFromAST();
    if (DeserializationListener)
      DeserializationListener->TypeRead(TypeIdx::fromTypeID(ID),
                                        TypesLoaded[Index]);
  }

  return TypesLoaded[Index].withFastQualifiers(FastQuals);
}

QualType ASTReader::getLocalType(ModuleFile &F, unsigned LocalID) {
  return GetType(getGlobalTypeID(F, LocalID));
}

serialization::TypeID 
ASTReader::getGlobalTypeID(ModuleFile &F, unsigned LocalID) const {
  unsigned FastQuals = LocalID & Qualifiers::FastMask;
  unsigned LocalIndex = LocalID >> Qualifiers::FastWidth;
  
  if (LocalIndex < NUM_PREDEF_TYPE_IDS)
    return LocalID;

  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = F.TypeRemap.find(LocalIndex - NUM_PREDEF_TYPE_IDS);
  assert(I != F.TypeRemap.end() && "Invalid index into type index remap");
  
  unsigned GlobalIndex = LocalIndex + I->second;
  return (GlobalIndex << Qualifiers::FastWidth) | FastQuals;
}

TemplateArgumentLocInfo
ASTReader::GetTemplateArgumentLocInfo(ModuleFile &F,
                                      TemplateArgument::ArgKind Kind,
                                      const RecordData &Record,
                                      unsigned &Index) {
  switch (Kind) {
  case TemplateArgument::Expression:
    return ReadExpr(F);
  case TemplateArgument::Type:
    return GetTypeSourceInfo(F, Record, Index);
  case TemplateArgument::Template: {
    NestedNameSpecifierLoc QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, 
                                                                     Index);
    SourceLocation TemplateNameLoc = ReadSourceLocation(F, Record, Index);
    return TemplateArgumentLocInfo(QualifierLoc, TemplateNameLoc,
                                   SourceLocation());
  }
  case TemplateArgument::TemplateExpansion: {
    NestedNameSpecifierLoc QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, 
                                                                     Index);
    SourceLocation TemplateNameLoc = ReadSourceLocation(F, Record, Index);
    SourceLocation EllipsisLoc = ReadSourceLocation(F, Record, Index);
    return TemplateArgumentLocInfo(QualifierLoc, TemplateNameLoc, 
                                   EllipsisLoc);
  }
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Declaration:
  case TemplateArgument::Pack:
    return TemplateArgumentLocInfo();
  }
  llvm_unreachable("unexpected template argument loc");
}

TemplateArgumentLoc
ASTReader::ReadTemplateArgumentLoc(ModuleFile &F,
                                   const RecordData &Record, unsigned &Index) {
  TemplateArgument Arg = ReadTemplateArgument(F, Record, Index);

  if (Arg.getKind() == TemplateArgument::Expression) {
    if (Record[Index++]) // bool InfoHasSameExpr.
      return TemplateArgumentLoc(Arg, TemplateArgumentLocInfo(Arg.getAsExpr()));
  }
  return TemplateArgumentLoc(Arg, GetTemplateArgumentLocInfo(F, Arg.getKind(),
                                                             Record, Index));
}

Decl *ASTReader::GetExternalDecl(uint32_t ID) {
  return GetDecl(ID);
}

uint64_t ASTReader::readCXXBaseSpecifiers(ModuleFile &M, const RecordData &Record, 
                                          unsigned &Idx){
  if (Idx >= Record.size())
    return 0;
  
  unsigned LocalID = Record[Idx++];
  return getGlobalBitOffset(M, M.CXXBaseSpecifiersOffsets[LocalID - 1]);
}

CXXBaseSpecifier *ASTReader::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  RecordLocation Loc = getLocalBitOffset(Offset);
  llvm::BitstreamCursor &Cursor = Loc.F->DeclsCursor;
  SavedStreamPosition SavedPosition(Cursor);
  Cursor.JumpToBit(Loc.Offset);
  ReadingKindTracker ReadingKind(Read_Decl, *this);
  RecordData Record;
  unsigned Code = Cursor.ReadCode();
  unsigned RecCode = Cursor.ReadRecord(Code, Record);
  if (RecCode != DECL_CXX_BASE_SPECIFIERS) {
    Error("Malformed AST file: missing C++ base specifiers");
    return 0;
  }

  unsigned Idx = 0;
  unsigned NumBases = Record[Idx++];
  void *Mem = Context.Allocate(sizeof(CXXBaseSpecifier) * NumBases);
  CXXBaseSpecifier *Bases = new (Mem) CXXBaseSpecifier [NumBases];
  for (unsigned I = 0; I != NumBases; ++I)
    Bases[I] = ReadCXXBaseSpecifier(*Loc.F, Record, Idx);
  return Bases;
}

serialization::DeclID 
ASTReader::getGlobalDeclID(ModuleFile &F, unsigned LocalID) const {
  if (LocalID < NUM_PREDEF_DECL_IDS)
    return LocalID;

  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = F.DeclRemap.find(LocalID - NUM_PREDEF_DECL_IDS);
  assert(I != F.DeclRemap.end() && "Invalid index into decl index remap");
  
  return LocalID + I->second;
}

bool ASTReader::isDeclIDFromModule(serialization::GlobalDeclID ID,
                                   ModuleFile &M) const {
  GlobalDeclMapType::const_iterator I = GlobalDeclMap.find(ID);
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  return &M == I->second;
}

ModuleFile *ASTReader::getOwningModuleFile(Decl *D) {
  if (!D->isFromASTFile())
    return 0;
  GlobalDeclMapType::const_iterator I = GlobalDeclMap.find(D->getGlobalID());
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  return I->second;
}

SourceLocation ASTReader::getSourceLocationForDeclID(GlobalDeclID ID) {
  if (ID < NUM_PREDEF_DECL_IDS)
    return SourceLocation();
  
  unsigned Index = ID - NUM_PREDEF_DECL_IDS;

  if (Index > DeclsLoaded.size()) {
    Error("declaration ID out-of-range for AST file");
    return SourceLocation();
  }
  
  if (Decl *D = DeclsLoaded[Index])
    return D->getLocation();

  unsigned RawLocation = 0;
  RecordLocation Rec = DeclCursorForID(ID, RawLocation);
  return ReadSourceLocation(*Rec.F, RawLocation);
}

Decl *ASTReader::GetDecl(DeclID ID) {
  if (ID < NUM_PREDEF_DECL_IDS) {    
    switch ((PredefinedDeclIDs)ID) {
    case PREDEF_DECL_NULL_ID:
      return 0;
        
    case PREDEF_DECL_TRANSLATION_UNIT_ID:
      return Context.getTranslationUnitDecl();
        
    case PREDEF_DECL_OBJC_ID_ID:
      return Context.getObjCIdDecl();

    case PREDEF_DECL_OBJC_SEL_ID:
      return Context.getObjCSelDecl();

    case PREDEF_DECL_OBJC_CLASS_ID:
      return Context.getObjCClassDecl();
        
    case PREDEF_DECL_OBJC_PROTOCOL_ID:
      return Context.getObjCProtocolDecl();
        
    case PREDEF_DECL_INT_128_ID:
      return Context.getInt128Decl();

    case PREDEF_DECL_UNSIGNED_INT_128_ID:
      return Context.getUInt128Decl();
        
    case PREDEF_DECL_OBJC_INSTANCETYPE_ID:
      return Context.getObjCInstanceTypeDecl();
    }
  }
  
  unsigned Index = ID - NUM_PREDEF_DECL_IDS;

  if (Index >= DeclsLoaded.size()) {
    Error("declaration ID out-of-range for AST file");
  }
  
  if (!DeclsLoaded[Index]) {
    ReadDeclRecord(ID);
    if (DeserializationListener)
      DeserializationListener->DeclRead(ID, DeclsLoaded[Index]);
  }

  return DeclsLoaded[Index];
}

DeclID ASTReader::mapGlobalIDToModuleFileGlobalID(ModuleFile &M, 
                                                  DeclID GlobalID) {
  if (GlobalID < NUM_PREDEF_DECL_IDS)
    return GlobalID;
  
  GlobalDeclMapType::const_iterator I = GlobalDeclMap.find(GlobalID);
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  ModuleFile *Owner = I->second;

  llvm::DenseMap<ModuleFile *, serialization::DeclID>::iterator Pos
    = M.GlobalToLocalDeclIDs.find(Owner);
  if (Pos == M.GlobalToLocalDeclIDs.end())
    return 0;
      
  return GlobalID - Owner->BaseDeclID + Pos->second;
}

serialization::DeclID ASTReader::ReadDeclID(ModuleFile &F, 
                                            const RecordData &Record,
                                            unsigned &Idx) {
  if (Idx >= Record.size()) {
    Error("Corrupted AST file");
    return 0;
  }
  
  return getGlobalDeclID(F, Record[Idx++]);
}

/// \brief Resolve the offset of a statement into a statement.
///
/// This operation will read a new statement from the external
/// source each time it is called, and is meant to be used via a
/// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
Stmt *ASTReader::GetExternalDeclStmt(uint64_t Offset) {
  // Switch case IDs are per Decl.
  ClearSwitchCaseIDs();

  // Offset here is a global offset across the entire chain.
  RecordLocation Loc = getLocalBitOffset(Offset);
  Loc.F->DeclsCursor.JumpToBit(Loc.Offset);
  return ReadStmtFromStream(*Loc.F);
}

namespace {
  class FindExternalLexicalDeclsVisitor {
    ASTReader &Reader;
    const DeclContext *DC;
    bool (*isKindWeWant)(Decl::Kind);
    
    SmallVectorImpl<Decl*> &Decls;
    bool PredefsVisited[NUM_PREDEF_DECL_IDS];

  public:
    FindExternalLexicalDeclsVisitor(ASTReader &Reader, const DeclContext *DC,
                                    bool (*isKindWeWant)(Decl::Kind),
                                    SmallVectorImpl<Decl*> &Decls)
      : Reader(Reader), DC(DC), isKindWeWant(isKindWeWant), Decls(Decls) 
    {
      for (unsigned I = 0; I != NUM_PREDEF_DECL_IDS; ++I)
        PredefsVisited[I] = false;
    }

    static bool visit(ModuleFile &M, bool Preorder, void *UserData) {
      if (Preorder)
        return false;

      FindExternalLexicalDeclsVisitor *This
        = static_cast<FindExternalLexicalDeclsVisitor *>(UserData);

      ModuleFile::DeclContextInfosMap::iterator Info
        = M.DeclContextInfos.find(This->DC);
      if (Info == M.DeclContextInfos.end() || !Info->second.LexicalDecls)
        return false;

      // Load all of the declaration IDs
      for (const KindDeclIDPair *ID = Info->second.LexicalDecls,
                               *IDE = ID + Info->second.NumLexicalDecls; 
           ID != IDE; ++ID) {
        if (This->isKindWeWant && !This->isKindWeWant((Decl::Kind)ID->first))
          continue;

        // Don't add predefined declarations to the lexical context more
        // than once.
        if (ID->second < NUM_PREDEF_DECL_IDS) {
          if (This->PredefsVisited[ID->second])
            continue;

          This->PredefsVisited[ID->second] = true;
        }

        if (Decl *D = This->Reader.GetLocalDecl(M, ID->second)) {
          if (!This->DC->isDeclInLexicalTraversal(D))
            This->Decls.push_back(D);
        }
      }

      return false;
    }
  };
}

ExternalLoadResult ASTReader::FindExternalLexicalDecls(const DeclContext *DC,
                                         bool (*isKindWeWant)(Decl::Kind),
                                         SmallVectorImpl<Decl*> &Decls) {
  // There might be lexical decls in multiple modules, for the TU at
  // least. Walk all of the modules in the order they were loaded.
  FindExternalLexicalDeclsVisitor Visitor(*this, DC, isKindWeWant, Decls);
  ModuleMgr.visitDepthFirst(&FindExternalLexicalDeclsVisitor::visit, &Visitor);
  ++NumLexicalDeclContextsRead;
  return ELR_Success;
}

namespace {

class DeclIDComp {
  ASTReader &Reader;
  ModuleFile &Mod;

public:
  DeclIDComp(ASTReader &Reader, ModuleFile &M) : Reader(Reader), Mod(M) {}

  bool operator()(LocalDeclID L, LocalDeclID R) const {
    SourceLocation LHS = getLocation(L);
    SourceLocation RHS = getLocation(R);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  bool operator()(SourceLocation LHS, LocalDeclID R) const {
    SourceLocation RHS = getLocation(R);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  bool operator()(LocalDeclID L, SourceLocation RHS) const {
    SourceLocation LHS = getLocation(L);
    return Reader.getSourceManager().isBeforeInTranslationUnit(LHS, RHS);
  }

  SourceLocation getLocation(LocalDeclID ID) const {
    return Reader.getSourceManager().getFileLoc(
            Reader.getSourceLocationForDeclID(Reader.getGlobalDeclID(Mod, ID)));
  }
};

}

void ASTReader::FindFileRegionDecls(FileID File,
                                    unsigned Offset, unsigned Length,
                                    SmallVectorImpl<Decl *> &Decls) {
  SourceManager &SM = getSourceManager();

  llvm::DenseMap<FileID, FileDeclsInfo>::iterator I = FileDeclIDs.find(File);
  if (I == FileDeclIDs.end())
    return;

  FileDeclsInfo &DInfo = I->second;
  if (DInfo.Decls.empty())
    return;

  SourceLocation
    BeginLoc = SM.getLocForStartOfFile(File).getLocWithOffset(Offset);
  SourceLocation EndLoc = BeginLoc.getLocWithOffset(Length);

  DeclIDComp DIDComp(*this, *DInfo.Mod);
  ArrayRef<serialization::LocalDeclID>::iterator
    BeginIt = std::lower_bound(DInfo.Decls.begin(), DInfo.Decls.end(),
                               BeginLoc, DIDComp);
  if (BeginIt != DInfo.Decls.begin())
    --BeginIt;

  // If we are pointing at a top-level decl inside an objc container, we need
  // to backtrack until we find it otherwise we will fail to report that the
  // region overlaps with an objc container.
  while (BeginIt != DInfo.Decls.begin() &&
         GetDecl(getGlobalDeclID(*DInfo.Mod, *BeginIt))
             ->isTopLevelDeclInObjCContainer())
    --BeginIt;

  ArrayRef<serialization::LocalDeclID>::iterator
    EndIt = std::upper_bound(DInfo.Decls.begin(), DInfo.Decls.end(),
                             EndLoc, DIDComp);
  if (EndIt != DInfo.Decls.end())
    ++EndIt;
  
  for (ArrayRef<serialization::LocalDeclID>::iterator
         DIt = BeginIt; DIt != EndIt; ++DIt)
    Decls.push_back(GetDecl(getGlobalDeclID(*DInfo.Mod, *DIt)));
}

namespace {
  /// \brief ModuleFile visitor used to perform name lookup into a
  /// declaration context.
  class DeclContextNameLookupVisitor {
    ASTReader &Reader;
    llvm::SmallVectorImpl<const DeclContext *> &Contexts;
    const DeclContext *DC;
    DeclarationName Name;
    SmallVectorImpl<NamedDecl *> &Decls;

  public:
    DeclContextNameLookupVisitor(ASTReader &Reader, 
                                 SmallVectorImpl<const DeclContext *> &Contexts, 
                                 DeclarationName Name,
                                 SmallVectorImpl<NamedDecl *> &Decls)
      : Reader(Reader), Contexts(Contexts), Name(Name), Decls(Decls) { }

    static bool visit(ModuleFile &M, void *UserData) {
      DeclContextNameLookupVisitor *This
        = static_cast<DeclContextNameLookupVisitor *>(UserData);

      // Check whether we have any visible declaration information for
      // this context in this module.
      ModuleFile::DeclContextInfosMap::iterator Info;
      bool FoundInfo = false;
      for (unsigned I = 0, N = This->Contexts.size(); I != N; ++I) {
        Info = M.DeclContextInfos.find(This->Contexts[I]);
        if (Info != M.DeclContextInfos.end() && 
            Info->second.NameLookupTableData) {
          FoundInfo = true;
          break;
        }
      }

      if (!FoundInfo)
        return false;
      
      // Look for this name within this module.
      ASTDeclContextNameLookupTable *LookupTable =
        Info->second.NameLookupTableData;
      ASTDeclContextNameLookupTable::iterator Pos
        = LookupTable->find(This->Name);
      if (Pos == LookupTable->end())
        return false;

      bool FoundAnything = false;
      ASTDeclContextNameLookupTrait::data_type Data = *Pos;
      for (; Data.first != Data.second; ++Data.first) {
        NamedDecl *ND = This->Reader.GetLocalDeclAs<NamedDecl>(M, *Data.first);
        if (!ND)
          continue;

        if (ND->getDeclName() != This->Name) {
          assert(!This->Name.getCXXNameType().isNull() && 
                 "Name mismatch without a type");
          continue;
        }
      
        // Record this declaration.
        FoundAnything = true;
        This->Decls.push_back(ND);
      }

      return FoundAnything;
    }
  };
}

DeclContext::lookup_result
ASTReader::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                          DeclarationName Name) {
  assert(DC->hasExternalVisibleStorage() &&
         "DeclContext has no visible decls in storage");
  if (!Name)
    return DeclContext::lookup_result(DeclContext::lookup_iterator(0),
                                      DeclContext::lookup_iterator(0));

  SmallVector<NamedDecl *, 64> Decls;
  
  // Compute the declaration contexts we need to look into. Multiple such
  // declaration contexts occur when two declaration contexts from disjoint
  // modules get merged, e.g., when two namespaces with the same name are 
  // independently defined in separate modules.
  SmallVector<const DeclContext *, 2> Contexts;
  Contexts.push_back(DC);
  
  if (DC->isNamespace()) {
    MergedDeclsMap::iterator Merged
      = MergedDecls.find(const_cast<Decl *>(cast<Decl>(DC)));
    if (Merged != MergedDecls.end()) {
      for (unsigned I = 0, N = Merged->second.size(); I != N; ++I)
        Contexts.push_back(cast<DeclContext>(GetDecl(Merged->second[I])));
    }
  }
  
  DeclContextNameLookupVisitor Visitor(*this, Contexts, Name, Decls);
  ModuleMgr.visit(&DeclContextNameLookupVisitor::visit, &Visitor);
  ++NumVisibleDeclContextsRead;
  SetExternalVisibleDeclsForName(DC, Name, Decls);
  return const_cast<DeclContext*>(DC)->lookup(Name);
}

namespace {
  /// \brief ModuleFile visitor used to retrieve all visible names in a
  /// declaration context.
  class DeclContextAllNamesVisitor {
    ASTReader &Reader;
    llvm::SmallVectorImpl<const DeclContext *> &Contexts;
    const DeclContext *DC;
    llvm::DenseMap<DeclarationName, SmallVector<NamedDecl *, 8> > &Decls;

  public:
    DeclContextAllNamesVisitor(ASTReader &Reader,
                               SmallVectorImpl<const DeclContext *> &Contexts,
                               llvm::DenseMap<DeclarationName,
                                           SmallVector<NamedDecl *, 8> > &Decls)
      : Reader(Reader), Contexts(Contexts), Decls(Decls) { }

    static bool visit(ModuleFile &M, void *UserData) {
      DeclContextAllNamesVisitor *This
        = static_cast<DeclContextAllNamesVisitor *>(UserData);

      // Check whether we have any visible declaration information for
      // this context in this module.
      ModuleFile::DeclContextInfosMap::iterator Info;
      bool FoundInfo = false;
      for (unsigned I = 0, N = This->Contexts.size(); I != N; ++I) {
        Info = M.DeclContextInfos.find(This->Contexts[I]);
        if (Info != M.DeclContextInfos.end() &&
            Info->second.NameLookupTableData) {
          FoundInfo = true;
          break;
        }
      }

      if (!FoundInfo)
        return false;

      ASTDeclContextNameLookupTable *LookupTable =
        Info->second.NameLookupTableData;
      bool FoundAnything = false;
      for (ASTDeclContextNameLookupTable::data_iterator
	     I = LookupTable->data_begin(), E = LookupTable->data_end();
	   I != E; ++I) {
        ASTDeclContextNameLookupTrait::data_type Data = *I;
        for (; Data.first != Data.second; ++Data.first) {
          NamedDecl *ND = This->Reader.GetLocalDeclAs<NamedDecl>(M,
                                                                 *Data.first);
          if (!ND)
            continue;

          // Record this declaration.
          FoundAnything = true;
          This->Decls[ND->getDeclName()].push_back(ND);
        }
      }

      return FoundAnything;
    }
  };
}

void ASTReader::completeVisibleDeclsMap(const DeclContext *DC) {
  if (!DC->hasExternalVisibleStorage())
    return;
  llvm::DenseMap<DeclarationName, llvm::SmallVector<NamedDecl*, 8> > Decls;

  // Compute the declaration contexts we need to look into. Multiple such
  // declaration contexts occur when two declaration contexts from disjoint
  // modules get merged, e.g., when two namespaces with the same name are
  // independently defined in separate modules.
  SmallVector<const DeclContext *, 2> Contexts;
  Contexts.push_back(DC);

  if (DC->isNamespace()) {
    MergedDeclsMap::iterator Merged
      = MergedDecls.find(const_cast<Decl *>(cast<Decl>(DC)));
    if (Merged != MergedDecls.end()) {
      for (unsigned I = 0, N = Merged->second.size(); I != N; ++I)
        Contexts.push_back(cast<DeclContext>(GetDecl(Merged->second[I])));
    }
  }

  DeclContextAllNamesVisitor Visitor(*this, Contexts, Decls);
  ModuleMgr.visit(&DeclContextAllNamesVisitor::visit, &Visitor);
  ++NumVisibleDeclContextsRead;

  for (llvm::DenseMap<DeclarationName,
                      llvm::SmallVector<NamedDecl*, 8> >::iterator
         I = Decls.begin(), E = Decls.end(); I != E; ++I) {
    SetExternalVisibleDeclsForName(DC, I->first, I->second);
  }
}

/// \brief Under non-PCH compilation the consumer receives the objc methods
/// before receiving the implementation, and codegen depends on this.
/// We simulate this by deserializing and passing to consumer the methods of the
/// implementation before passing the deserialized implementation decl.
static void PassObjCImplDeclToConsumer(ObjCImplDecl *ImplD,
                                       ASTConsumer *Consumer) {
  assert(ImplD && Consumer);

  for (ObjCImplDecl::method_iterator
         I = ImplD->meth_begin(), E = ImplD->meth_end(); I != E; ++I)
    Consumer->HandleInterestingDecl(DeclGroupRef(*I));

  Consumer->HandleInterestingDecl(DeclGroupRef(ImplD));
}

void ASTReader::PassInterestingDeclsToConsumer() {
  assert(Consumer);
  while (!InterestingDecls.empty()) {
    Decl *D = InterestingDecls.front();
    InterestingDecls.pop_front();

    PassInterestingDeclToConsumer(D);
  }
}

void ASTReader::PassInterestingDeclToConsumer(Decl *D) {
  if (ObjCImplDecl *ImplD = dyn_cast<ObjCImplDecl>(D))
    PassObjCImplDeclToConsumer(ImplD, Consumer);
  else
    Consumer->HandleInterestingDecl(DeclGroupRef(D));
}

void ASTReader::StartTranslationUnit(ASTConsumer *Consumer) {
  this->Consumer = Consumer;

  if (!Consumer)
    return;

  for (unsigned I = 0, N = ExternalDefinitions.size(); I != N; ++I) {
    // Force deserialization of this decl, which will cause it to be queued for
    // passing to the consumer.
    GetDecl(ExternalDefinitions[I]);
  }
  ExternalDefinitions.clear();

  PassInterestingDeclsToConsumer();
}

void ASTReader::PrintStats() {
  std::fprintf(stderr, "*** AST File Statistics:\n");

  unsigned NumTypesLoaded
    = TypesLoaded.size() - std::count(TypesLoaded.begin(), TypesLoaded.end(),
                                      QualType());
  unsigned NumDeclsLoaded
    = DeclsLoaded.size() - std::count(DeclsLoaded.begin(), DeclsLoaded.end(),
                                      (Decl *)0);
  unsigned NumIdentifiersLoaded
    = IdentifiersLoaded.size() - std::count(IdentifiersLoaded.begin(),
                                            IdentifiersLoaded.end(),
                                            (IdentifierInfo *)0);
  unsigned NumSelectorsLoaded
    = SelectorsLoaded.size() - std::count(SelectorsLoaded.begin(),
                                          SelectorsLoaded.end(),
                                          Selector());

  std::fprintf(stderr, "  %u stat cache hits\n", NumStatHits);
  std::fprintf(stderr, "  %u stat cache misses\n", NumStatMisses);
  if (unsigned TotalNumSLocEntries = getTotalNumSLocs())
    std::fprintf(stderr, "  %u/%u source location entries read (%f%%)\n",
                 NumSLocEntriesRead, TotalNumSLocEntries,
                 ((float)NumSLocEntriesRead/TotalNumSLocEntries * 100));
  if (!TypesLoaded.empty())
    std::fprintf(stderr, "  %u/%u types read (%f%%)\n",
                 NumTypesLoaded, (unsigned)TypesLoaded.size(),
                 ((float)NumTypesLoaded/TypesLoaded.size() * 100));
  if (!DeclsLoaded.empty())
    std::fprintf(stderr, "  %u/%u declarations read (%f%%)\n",
                 NumDeclsLoaded, (unsigned)DeclsLoaded.size(),
                 ((float)NumDeclsLoaded/DeclsLoaded.size() * 100));
  if (!IdentifiersLoaded.empty())
    std::fprintf(stderr, "  %u/%u identifiers read (%f%%)\n",
                 NumIdentifiersLoaded, (unsigned)IdentifiersLoaded.size(),
                 ((float)NumIdentifiersLoaded/IdentifiersLoaded.size() * 100));
  if (!SelectorsLoaded.empty())
    std::fprintf(stderr, "  %u/%u selectors read (%f%%)\n",
                 NumSelectorsLoaded, (unsigned)SelectorsLoaded.size(),
                 ((float)NumSelectorsLoaded/SelectorsLoaded.size() * 100));
  if (TotalNumStatements)
    std::fprintf(stderr, "  %u/%u statements read (%f%%)\n",
                 NumStatementsRead, TotalNumStatements,
                 ((float)NumStatementsRead/TotalNumStatements * 100));
  if (TotalNumMacros)
    std::fprintf(stderr, "  %u/%u macros read (%f%%)\n",
                 NumMacrosRead, TotalNumMacros,
                 ((float)NumMacrosRead/TotalNumMacros * 100));
  if (TotalLexicalDeclContexts)
    std::fprintf(stderr, "  %u/%u lexical declcontexts read (%f%%)\n",
                 NumLexicalDeclContextsRead, TotalLexicalDeclContexts,
                 ((float)NumLexicalDeclContextsRead/TotalLexicalDeclContexts
                  * 100));
  if (TotalVisibleDeclContexts)
    std::fprintf(stderr, "  %u/%u visible declcontexts read (%f%%)\n",
                 NumVisibleDeclContextsRead, TotalVisibleDeclContexts,
                 ((float)NumVisibleDeclContextsRead/TotalVisibleDeclContexts
                  * 100));
  if (TotalNumMethodPoolEntries) {
    std::fprintf(stderr, "  %u/%u method pool entries read (%f%%)\n",
                 NumMethodPoolEntriesRead, TotalNumMethodPoolEntries,
                 ((float)NumMethodPoolEntriesRead/TotalNumMethodPoolEntries
                  * 100));
    std::fprintf(stderr, "  %u method pool misses\n", NumMethodPoolMisses);
  }
  std::fprintf(stderr, "\n");
  dump();
  std::fprintf(stderr, "\n");
}

template<typename Key, typename ModuleFile, unsigned InitialCapacity>
static void 
dumpModuleIDMap(StringRef Name,
                const ContinuousRangeMap<Key, ModuleFile *, 
                                         InitialCapacity> &Map) {
  if (Map.begin() == Map.end())
    return;
  
  typedef ContinuousRangeMap<Key, ModuleFile *, InitialCapacity> MapType;
  llvm::errs() << Name << ":\n";
  for (typename MapType::const_iterator I = Map.begin(), IEnd = Map.end(); 
       I != IEnd; ++I) {
    llvm::errs() << "  " << I->first << " -> " << I->second->FileName
      << "\n";
  }
}

void ASTReader::dump() {
  llvm::errs() << "*** PCH/ModuleFile Remappings:\n";
  dumpModuleIDMap("Global bit offset map", GlobalBitOffsetsMap);
  dumpModuleIDMap("Global source location entry map", GlobalSLocEntryMap);
  dumpModuleIDMap("Global type map", GlobalTypeMap);
  dumpModuleIDMap("Global declaration map", GlobalDeclMap);
  dumpModuleIDMap("Global identifier map", GlobalIdentifierMap);
  dumpModuleIDMap("Global submodule map", GlobalSubmoduleMap);
  dumpModuleIDMap("Global selector map", GlobalSelectorMap);
  dumpModuleIDMap("Global preprocessed entity map", 
                  GlobalPreprocessedEntityMap);
  
  llvm::errs() << "\n*** PCH/Modules Loaded:";
  for (ModuleManager::ModuleConstIterator M = ModuleMgr.begin(), 
                                       MEnd = ModuleMgr.end();
       M != MEnd; ++M)
    (*M)->dump();
}

/// Return the amount of memory used by memory buffers, breaking down
/// by heap-backed versus mmap'ed memory.
void ASTReader::getMemoryBufferSizes(MemoryBufferSizes &sizes) const {
  for (ModuleConstIterator I = ModuleMgr.begin(),
      E = ModuleMgr.end(); I != E; ++I) {
    if (llvm::MemoryBuffer *buf = (*I)->Buffer.get()) {
      size_t bytes = buf->getBufferSize();
      switch (buf->getBufferKind()) {
        case llvm::MemoryBuffer::MemoryBuffer_Malloc:
          sizes.malloc_bytes += bytes;
          break;
        case llvm::MemoryBuffer::MemoryBuffer_MMap:
          sizes.mmap_bytes += bytes;
          break;
      }
    }
  }
}

void ASTReader::InitializeSema(Sema &S) {
  SemaObj = &S;
  S.ExternalSource = this;

  // Makes sure any declarations that were deserialized "too early"
  // still get added to the identifier's declaration chains.
  for (unsigned I = 0, N = PreloadedDecls.size(); I != N; ++I) {
    SemaObj->pushExternalDeclIntoScope(PreloadedDecls[I], 
                                       PreloadedDecls[I]->getDeclName());
  }
  PreloadedDecls.clear();

  // Load the offsets of the declarations that Sema references.
  // They will be lazily deserialized when needed.
  if (!SemaDeclRefs.empty()) {
    assert(SemaDeclRefs.size() == 2 && "More decl refs than expected!");
    if (!SemaObj->StdNamespace)
      SemaObj->StdNamespace = SemaDeclRefs[0];
    if (!SemaObj->StdBadAlloc)
      SemaObj->StdBadAlloc = SemaDeclRefs[1];
  }

  if (!FPPragmaOptions.empty()) {
    assert(FPPragmaOptions.size() == 1 && "Wrong number of FP_PRAGMA_OPTIONS");
    SemaObj->FPFeatures.fp_contract = FPPragmaOptions[0];
  }

  if (!OpenCLExtensions.empty()) {
    unsigned I = 0;
#define OPENCLEXT(nm)  SemaObj->OpenCLFeatures.nm = OpenCLExtensions[I++];
#include "clang/Basic/OpenCLExtensions.def"

    assert(OpenCLExtensions.size() == I && "Wrong number of OPENCL_EXTENSIONS");
  }
}

IdentifierInfo* ASTReader::get(const char *NameStart, const char *NameEnd) {
  IdentifierLookupVisitor Visitor(StringRef(NameStart, NameEnd - NameStart),
                                  /*PriorGeneration=*/0);
  ModuleMgr.visit(IdentifierLookupVisitor::visit, &Visitor);
  IdentifierInfo *II = Visitor.getIdentifierInfo();
  markIdentifierUpToDate(II);
  return II;
}

namespace clang {
  /// \brief An identifier-lookup iterator that enumerates all of the
  /// identifiers stored within a set of AST files.
  class ASTIdentifierIterator : public IdentifierIterator {
    /// \brief The AST reader whose identifiers are being enumerated.
    const ASTReader &Reader;

    /// \brief The current index into the chain of AST files stored in
    /// the AST reader.
    unsigned Index;

    /// \brief The current position within the identifier lookup table
    /// of the current AST file.
    ASTIdentifierLookupTable::key_iterator Current;

    /// \brief The end position within the identifier lookup table of
    /// the current AST file.
    ASTIdentifierLookupTable::key_iterator End;

  public:
    explicit ASTIdentifierIterator(const ASTReader &Reader);

    virtual StringRef Next();
  };
}

ASTIdentifierIterator::ASTIdentifierIterator(const ASTReader &Reader)
  : Reader(Reader), Index(Reader.ModuleMgr.size() - 1) {
  ASTIdentifierLookupTable *IdTable
    = (ASTIdentifierLookupTable *)Reader.ModuleMgr[Index].IdentifierLookupTable;
  Current = IdTable->key_begin();
  End = IdTable->key_end();
}

StringRef ASTIdentifierIterator::Next() {
  while (Current == End) {
    // If we have exhausted all of our AST files, we're done.
    if (Index == 0)
      return StringRef();

    --Index;
    ASTIdentifierLookupTable *IdTable
      = (ASTIdentifierLookupTable *)Reader.ModuleMgr[Index].
        IdentifierLookupTable;
    Current = IdTable->key_begin();
    End = IdTable->key_end();
  }

  // We have any identifiers remaining in the current AST file; return
  // the next one.
  std::pair<const char*, unsigned> Key = *Current;
  ++Current;
  return StringRef(Key.first, Key.second);
}

IdentifierIterator *ASTReader::getIdentifiers() const {
  return new ASTIdentifierIterator(*this);
}

namespace clang { namespace serialization {
  class ReadMethodPoolVisitor {
    ASTReader &Reader;
    Selector Sel;
    unsigned PriorGeneration;
    llvm::SmallVector<ObjCMethodDecl *, 4> InstanceMethods;
    llvm::SmallVector<ObjCMethodDecl *, 4> FactoryMethods;

  public:
    ReadMethodPoolVisitor(ASTReader &Reader, Selector Sel, 
                          unsigned PriorGeneration)
      : Reader(Reader), Sel(Sel), PriorGeneration(PriorGeneration) { }
    
    static bool visit(ModuleFile &M, void *UserData) {
      ReadMethodPoolVisitor *This
        = static_cast<ReadMethodPoolVisitor *>(UserData);
      
      if (!M.SelectorLookupTable)
        return false;
      
      // If we've already searched this module file, skip it now.
      if (M.Generation <= This->PriorGeneration)
        return true;

      ASTSelectorLookupTable *PoolTable
        = (ASTSelectorLookupTable*)M.SelectorLookupTable;
      ASTSelectorLookupTable::iterator Pos = PoolTable->find(This->Sel);
      if (Pos == PoolTable->end())
        return false;
      
      ++This->Reader.NumSelectorsRead;
      // FIXME: Not quite happy with the statistics here. We probably should
      // disable this tracking when called via LoadSelector.
      // Also, should entries without methods count as misses?
      ++This->Reader.NumMethodPoolEntriesRead;
      ASTSelectorLookupTrait::data_type Data = *Pos;
      if (This->Reader.DeserializationListener)
        This->Reader.DeserializationListener->SelectorRead(Data.ID, 
                                                           This->Sel);
      
      This->InstanceMethods.append(Data.Instance.begin(), Data.Instance.end());
      This->FactoryMethods.append(Data.Factory.begin(), Data.Factory.end());
      return true;
    }
    
    /// \brief Retrieve the instance methods found by this visitor.
    ArrayRef<ObjCMethodDecl *> getInstanceMethods() const { 
      return InstanceMethods; 
    }

    /// \brief Retrieve the instance methods found by this visitor.
    ArrayRef<ObjCMethodDecl *> getFactoryMethods() const { 
      return FactoryMethods;
    }
  };
} } // end namespace clang::serialization

/// \brief Add the given set of methods to the method list.
static void addMethodsToPool(Sema &S, ArrayRef<ObjCMethodDecl *> Methods,
                             ObjCMethodList &List) {
  for (unsigned I = 0, N = Methods.size(); I != N; ++I) {
    S.addMethodToGlobalList(&List, Methods[I]);
  }
}
                             
void ASTReader::ReadMethodPool(Selector Sel) {
  // Get the selector generation and update it to the current generation.
  unsigned &Generation = SelectorGeneration[Sel];
  unsigned PriorGeneration = Generation;
  Generation = CurrentGeneration;
  
  // Search for methods defined with this selector.
  ReadMethodPoolVisitor Visitor(*this, Sel, PriorGeneration);
  ModuleMgr.visit(&ReadMethodPoolVisitor::visit, &Visitor);
  
  if (Visitor.getInstanceMethods().empty() &&
      Visitor.getFactoryMethods().empty()) {
    ++NumMethodPoolMisses;
    return;
  }
  
  if (!getSema())
    return;
  
  Sema &S = *getSema();
  Sema::GlobalMethodPool::iterator Pos
    = S.MethodPool.insert(std::make_pair(Sel, Sema::GlobalMethods())).first;
  
  addMethodsToPool(S, Visitor.getInstanceMethods(), Pos->second.first);
  addMethodsToPool(S, Visitor.getFactoryMethods(), Pos->second.second);
}

void ASTReader::ReadKnownNamespaces(
                          SmallVectorImpl<NamespaceDecl *> &Namespaces) {
  Namespaces.clear();
  
  for (unsigned I = 0, N = KnownNamespaces.size(); I != N; ++I) {
    if (NamespaceDecl *Namespace 
                = dyn_cast_or_null<NamespaceDecl>(GetDecl(KnownNamespaces[I])))
      Namespaces.push_back(Namespace);
  }
}

void ASTReader::ReadTentativeDefinitions(
                  SmallVectorImpl<VarDecl *> &TentativeDefs) {
  for (unsigned I = 0, N = TentativeDefinitions.size(); I != N; ++I) {
    VarDecl *Var = dyn_cast_or_null<VarDecl>(GetDecl(TentativeDefinitions[I]));
    if (Var)
      TentativeDefs.push_back(Var);
  }
  TentativeDefinitions.clear();
}

void ASTReader::ReadUnusedFileScopedDecls(
                               SmallVectorImpl<const DeclaratorDecl *> &Decls) {
  for (unsigned I = 0, N = UnusedFileScopedDecls.size(); I != N; ++I) {
    DeclaratorDecl *D
      = dyn_cast_or_null<DeclaratorDecl>(GetDecl(UnusedFileScopedDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  UnusedFileScopedDecls.clear();
}

void ASTReader::ReadDelegatingConstructors(
                                 SmallVectorImpl<CXXConstructorDecl *> &Decls) {
  for (unsigned I = 0, N = DelegatingCtorDecls.size(); I != N; ++I) {
    CXXConstructorDecl *D
      = dyn_cast_or_null<CXXConstructorDecl>(GetDecl(DelegatingCtorDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  DelegatingCtorDecls.clear();
}

void ASTReader::ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls) {
  for (unsigned I = 0, N = ExtVectorDecls.size(); I != N; ++I) {
    TypedefNameDecl *D
      = dyn_cast_or_null<TypedefNameDecl>(GetDecl(ExtVectorDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  ExtVectorDecls.clear();
}

void ASTReader::ReadDynamicClasses(SmallVectorImpl<CXXRecordDecl *> &Decls) {
  for (unsigned I = 0, N = DynamicClasses.size(); I != N; ++I) {
    CXXRecordDecl *D
      = dyn_cast_or_null<CXXRecordDecl>(GetDecl(DynamicClasses[I]));
    if (D)
      Decls.push_back(D);
  }
  DynamicClasses.clear();
}

void 
ASTReader::ReadLocallyScopedExternalDecls(SmallVectorImpl<NamedDecl *> &Decls) {
  for (unsigned I = 0, N = LocallyScopedExternalDecls.size(); I != N; ++I) {
    NamedDecl *D 
      = dyn_cast_or_null<NamedDecl>(GetDecl(LocallyScopedExternalDecls[I]));
    if (D)
      Decls.push_back(D);
  }
  LocallyScopedExternalDecls.clear();
}

void ASTReader::ReadReferencedSelectors(
       SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels) {
  if (ReferencedSelectorsData.empty())
    return;
  
  // If there are @selector references added them to its pool. This is for
  // implementation of -Wselector.
  unsigned int DataSize = ReferencedSelectorsData.size()-1;
  unsigned I = 0;
  while (I < DataSize) {
    Selector Sel = DecodeSelector(ReferencedSelectorsData[I++]);
    SourceLocation SelLoc
      = SourceLocation::getFromRawEncoding(ReferencedSelectorsData[I++]);
    Sels.push_back(std::make_pair(Sel, SelLoc));
  }
  ReferencedSelectorsData.clear();
}

void ASTReader::ReadWeakUndeclaredIdentifiers(
       SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WeakIDs) {
  if (WeakUndeclaredIdentifiers.empty())
    return;

  for (unsigned I = 0, N = WeakUndeclaredIdentifiers.size(); I < N; /*none*/) {
    IdentifierInfo *WeakId 
      = DecodeIdentifierInfo(WeakUndeclaredIdentifiers[I++]);
    IdentifierInfo *AliasId 
      = DecodeIdentifierInfo(WeakUndeclaredIdentifiers[I++]);
    SourceLocation Loc
      = SourceLocation::getFromRawEncoding(WeakUndeclaredIdentifiers[I++]);
    bool Used = WeakUndeclaredIdentifiers[I++];
    WeakInfo WI(AliasId, Loc);
    WI.setUsed(Used);
    WeakIDs.push_back(std::make_pair(WeakId, WI));
  }
  WeakUndeclaredIdentifiers.clear();
}

void ASTReader::ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables) {
  for (unsigned Idx = 0, N = VTableUses.size(); Idx < N; /* In loop */) {
    ExternalVTableUse VT;
    VT.Record = dyn_cast_or_null<CXXRecordDecl>(GetDecl(VTableUses[Idx++]));
    VT.Location = SourceLocation::getFromRawEncoding(VTableUses[Idx++]);
    VT.DefinitionRequired = VTableUses[Idx++];
    VTables.push_back(VT);
  }
  
  VTableUses.clear();
}

void ASTReader::ReadPendingInstantiations(
       SmallVectorImpl<std::pair<ValueDecl *, SourceLocation> > &Pending) {
  for (unsigned Idx = 0, N = PendingInstantiations.size(); Idx < N;) {
    ValueDecl *D = cast<ValueDecl>(GetDecl(PendingInstantiations[Idx++]));
    SourceLocation Loc
      = SourceLocation::getFromRawEncoding(PendingInstantiations[Idx++]);
    Pending.push_back(std::make_pair(D, Loc));
  }  
  PendingInstantiations.clear();
}

void ASTReader::LoadSelector(Selector Sel) {
  // It would be complicated to avoid reading the methods anyway. So don't.
  ReadMethodPool(Sel);
}

void ASTReader::SetIdentifierInfo(IdentifierID ID, IdentifierInfo *II) {
  assert(ID && "Non-zero identifier ID required");
  assert(ID <= IdentifiersLoaded.size() && "identifier ID out of range");
  IdentifiersLoaded[ID - 1] = II;
  if (DeserializationListener)
    DeserializationListener->IdentifierRead(ID, II);
}

/// \brief Set the globally-visible declarations associated with the given
/// identifier.
///
/// If the AST reader is currently in a state where the given declaration IDs
/// cannot safely be resolved, they are queued until it is safe to resolve
/// them.
///
/// \param II an IdentifierInfo that refers to one or more globally-visible
/// declarations.
///
/// \param DeclIDs the set of declaration IDs with the name @p II that are
/// visible at global scope.
///
/// \param Nonrecursive should be true to indicate that the caller knows that
/// this call is non-recursive, and therefore the globally-visible declarations
/// will not be placed onto the pending queue.
void
ASTReader::SetGloballyVisibleDecls(IdentifierInfo *II,
                              const SmallVectorImpl<uint32_t> &DeclIDs,
                                   bool Nonrecursive) {
  if (NumCurrentElementsDeserializing && !Nonrecursive) {
    PendingIdentifierInfos.push_back(PendingIdentifierInfo());
    PendingIdentifierInfo &PII = PendingIdentifierInfos.back();
    PII.II = II;
    PII.DeclIDs.append(DeclIDs.begin(), DeclIDs.end());
    return;
  }

  for (unsigned I = 0, N = DeclIDs.size(); I != N; ++I) {
    NamedDecl *D = cast<NamedDecl>(GetDecl(DeclIDs[I]));
    if (SemaObj) {
      // Introduce this declaration into the translation-unit scope
      // and add it to the declaration chain for this identifier, so
      // that (unqualified) name lookup will find it.
      SemaObj->pushExternalDeclIntoScope(D, II);
    } else {
      // Queue this declaration so that it will be added to the
      // translation unit scope and identifier's declaration chain
      // once a Sema object is known.
      PreloadedDecls.push_back(D);
    }
  }
}

IdentifierInfo *ASTReader::DecodeIdentifierInfo(IdentifierID ID) {
  if (ID == 0)
    return 0;

  if (IdentifiersLoaded.empty()) {
    Error("no identifier table in AST file");
    return 0;
  }

  ID -= 1;
  if (!IdentifiersLoaded[ID]) {
    GlobalIdentifierMapType::iterator I = GlobalIdentifierMap.find(ID + 1);
    assert(I != GlobalIdentifierMap.end() && "Corrupted global identifier map");
    ModuleFile *M = I->second;
    unsigned Index = ID - M->BaseIdentifierID;
    const char *Str = M->IdentifierTableData + M->IdentifierOffsets[Index];

    // All of the strings in the AST file are preceded by a 16-bit length.
    // Extract that 16-bit length to avoid having to execute strlen().
    // NOTE: 'StrLenPtr' is an 'unsigned char*' so that we load bytes as
    //  unsigned integers.  This is important to avoid integer overflow when
    //  we cast them to 'unsigned'.
    const unsigned char *StrLenPtr = (const unsigned char*) Str - 2;
    unsigned StrLen = (((unsigned) StrLenPtr[0])
                       | (((unsigned) StrLenPtr[1]) << 8)) - 1;
    IdentifiersLoaded[ID]
      = &PP.getIdentifierTable().get(StringRef(Str, StrLen));
    if (DeserializationListener)
      DeserializationListener->IdentifierRead(ID + 1, IdentifiersLoaded[ID]);
  }

  return IdentifiersLoaded[ID];
}

IdentifierInfo *ASTReader::getLocalIdentifier(ModuleFile &M, unsigned LocalID) {
  return DecodeIdentifierInfo(getGlobalIdentifierID(M, LocalID));
}

IdentifierID ASTReader::getGlobalIdentifierID(ModuleFile &M, unsigned LocalID) {
  if (LocalID < NUM_PREDEF_IDENT_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.IdentifierRemap.find(LocalID - NUM_PREDEF_IDENT_IDS);
  assert(I != M.IdentifierRemap.end() 
         && "Invalid index into identifier index remap");
  
  return LocalID + I->second;
}

bool ASTReader::ReadSLocEntry(int ID) {
  return ReadSLocEntryRecord(ID) != Success;
}

serialization::SubmoduleID
ASTReader::getGlobalSubmoduleID(ModuleFile &M, unsigned LocalID) {
  if (LocalID < NUM_PREDEF_SUBMODULE_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.SubmoduleRemap.find(LocalID - NUM_PREDEF_SUBMODULE_IDS);
  assert(I != M.SubmoduleRemap.end() 
         && "Invalid index into identifier index remap");
  
  return LocalID + I->second;
}

Module *ASTReader::getSubmodule(SubmoduleID GlobalID) {
  if (GlobalID < NUM_PREDEF_SUBMODULE_IDS) {
    assert(GlobalID == 0 && "Unhandled global submodule ID");
    return 0;
  }
  
  if (GlobalID > SubmodulesLoaded.size()) {
    Error("submodule ID out of range in AST file");
    return 0;
  }
  
  return SubmodulesLoaded[GlobalID - NUM_PREDEF_SUBMODULE_IDS];
}
                               
Selector ASTReader::getLocalSelector(ModuleFile &M, unsigned LocalID) {
  return DecodeSelector(getGlobalSelectorID(M, LocalID));
}

Selector ASTReader::DecodeSelector(serialization::SelectorID ID) {
  if (ID == 0)
    return Selector();

  if (ID > SelectorsLoaded.size()) {
    Error("selector ID out of range in AST file");
    return Selector();
  }

  if (SelectorsLoaded[ID - 1].getAsOpaquePtr() == 0) {
    // Load this selector from the selector table.
    GlobalSelectorMapType::iterator I = GlobalSelectorMap.find(ID);
    assert(I != GlobalSelectorMap.end() && "Corrupted global selector map");
    ModuleFile &M = *I->second;
    ASTSelectorLookupTrait Trait(*this, M);
    unsigned Idx = ID - M.BaseSelectorID - NUM_PREDEF_SELECTOR_IDS;
    SelectorsLoaded[ID - 1] =
      Trait.ReadKey(M.SelectorLookupTableData + M.SelectorOffsets[Idx], 0);
    if (DeserializationListener)
      DeserializationListener->SelectorRead(ID, SelectorsLoaded[ID - 1]);
  }

  return SelectorsLoaded[ID - 1];
}

Selector ASTReader::GetExternalSelector(serialization::SelectorID ID) {
  return DecodeSelector(ID);
}

uint32_t ASTReader::GetNumExternalSelectors() {
  // ID 0 (the null selector) is considered an external selector.
  return getTotalNumSelectors() + 1;
}

serialization::SelectorID
ASTReader::getGlobalSelectorID(ModuleFile &M, unsigned LocalID) const {
  if (LocalID < NUM_PREDEF_SELECTOR_IDS)
    return LocalID;
  
  ContinuousRangeMap<uint32_t, int, 2>::iterator I
    = M.SelectorRemap.find(LocalID - NUM_PREDEF_SELECTOR_IDS);
  assert(I != M.SelectorRemap.end() 
         && "Invalid index into identifier index remap");
  
  return LocalID + I->second;
}

DeclarationName
ASTReader::ReadDeclarationName(ModuleFile &F, 
                               const RecordData &Record, unsigned &Idx) {
  DeclarationName::NameKind Kind = (DeclarationName::NameKind)Record[Idx++];
  switch (Kind) {
  case DeclarationName::Identifier:
    return DeclarationName(GetIdentifierInfo(F, Record, Idx));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return DeclarationName(ReadSelector(F, Record, Idx));

  case DeclarationName::CXXConstructorName:
    return Context.DeclarationNames.getCXXConstructorName(
                          Context.getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXDestructorName:
    return Context.DeclarationNames.getCXXDestructorName(
                          Context.getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXConversionFunctionName:
    return Context.DeclarationNames.getCXXConversionFunctionName(
                          Context.getCanonicalType(readType(F, Record, Idx)));

  case DeclarationName::CXXOperatorName:
    return Context.DeclarationNames.getCXXOperatorName(
                                       (OverloadedOperatorKind)Record[Idx++]);

  case DeclarationName::CXXLiteralOperatorName:
    return Context.DeclarationNames.getCXXLiteralOperatorName(
                                       GetIdentifierInfo(F, Record, Idx));

  case DeclarationName::CXXUsingDirective:
    return DeclarationName::getUsingDirectiveName();
  }

  llvm_unreachable("Invalid NameKind!");
}

void ASTReader::ReadDeclarationNameLoc(ModuleFile &F,
                                       DeclarationNameLoc &DNLoc,
                                       DeclarationName Name,
                                      const RecordData &Record, unsigned &Idx) {
  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    DNLoc.NamedType.TInfo = GetTypeSourceInfo(F, Record, Idx);
    break;

  case DeclarationName::CXXOperatorName:
    DNLoc.CXXOperatorName.BeginOpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    DNLoc.CXXOperatorName.EndOpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    break;

  case DeclarationName::CXXLiteralOperatorName:
    DNLoc.CXXLiteralOperatorName.OpNameLoc
        = ReadSourceLocation(F, Record, Idx).getRawEncoding();
    break;

  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
    break;
  }
}

void ASTReader::ReadDeclarationNameInfo(ModuleFile &F,
                                        DeclarationNameInfo &NameInfo,
                                      const RecordData &Record, unsigned &Idx) {
  NameInfo.setName(ReadDeclarationName(F, Record, Idx));
  NameInfo.setLoc(ReadSourceLocation(F, Record, Idx));
  DeclarationNameLoc DNLoc;
  ReadDeclarationNameLoc(F, DNLoc, NameInfo.getName(), Record, Idx);
  NameInfo.setInfo(DNLoc);
}

void ASTReader::ReadQualifierInfo(ModuleFile &F, QualifierInfo &Info,
                                  const RecordData &Record, unsigned &Idx) {
  Info.QualifierLoc = ReadNestedNameSpecifierLoc(F, Record, Idx);
  unsigned NumTPLists = Record[Idx++];
  Info.NumTemplParamLists = NumTPLists;
  if (NumTPLists) {
    Info.TemplParamLists = new (Context) TemplateParameterList*[NumTPLists];
    for (unsigned i=0; i != NumTPLists; ++i)
      Info.TemplParamLists[i] = ReadTemplateParameterList(F, Record, Idx);
  }
}

TemplateName
ASTReader::ReadTemplateName(ModuleFile &F, const RecordData &Record, 
                            unsigned &Idx) {
  TemplateName::NameKind Kind = (TemplateName::NameKind)Record[Idx++];
  switch (Kind) {
  case TemplateName::Template:
      return TemplateName(ReadDeclAs<TemplateDecl>(F, Record, Idx));

  case TemplateName::OverloadedTemplate: {
    unsigned size = Record[Idx++];
    UnresolvedSet<8> Decls;
    while (size--)
      Decls.addDecl(ReadDeclAs<NamedDecl>(F, Record, Idx));

    return Context.getOverloadedTemplateName(Decls.begin(), Decls.end());
  }

  case TemplateName::QualifiedTemplate: {
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(F, Record, Idx);
    bool hasTemplKeyword = Record[Idx++];
    TemplateDecl *Template = ReadDeclAs<TemplateDecl>(F, Record, Idx);
    return Context.getQualifiedTemplateName(NNS, hasTemplKeyword, Template);
  }

  case TemplateName::DependentTemplate: {
    NestedNameSpecifier *NNS = ReadNestedNameSpecifier(F, Record, Idx);
    if (Record[Idx++])  // isIdentifier
      return Context.getDependentTemplateName(NNS,
                                               GetIdentifierInfo(F, Record, 
                                                                 Idx));
    return Context.getDependentTemplateName(NNS,
                                         (OverloadedOperatorKind)Record[Idx++]);
  }

  case TemplateName::SubstTemplateTemplateParm: {
    TemplateTemplateParmDecl *param
      = ReadDeclAs<TemplateTemplateParmDecl>(F, Record, Idx);
    if (!param) return TemplateName();
    TemplateName replacement = ReadTemplateName(F, Record, Idx);
    return Context.getSubstTemplateTemplateParm(param, replacement);
  }
      
  case TemplateName::SubstTemplateTemplateParmPack: {
    TemplateTemplateParmDecl *Param 
      = ReadDeclAs<TemplateTemplateParmDecl>(F, Record, Idx);
    if (!Param)
      return TemplateName();
    
    TemplateArgument ArgPack = ReadTemplateArgument(F, Record, Idx);
    if (ArgPack.getKind() != TemplateArgument::Pack)
      return TemplateName();
    
    return Context.getSubstTemplateTemplateParmPack(Param, ArgPack);
  }
  }

  llvm_unreachable("Unhandled template name kind!");
}

TemplateArgument
ASTReader::ReadTemplateArgument(ModuleFile &F,
                                const RecordData &Record, unsigned &Idx) {
  TemplateArgument::ArgKind Kind = (TemplateArgument::ArgKind)Record[Idx++];
  switch (Kind) {
  case TemplateArgument::Null:
    return TemplateArgument();
  case TemplateArgument::Type:
    return TemplateArgument(readType(F, Record, Idx));
  case TemplateArgument::Declaration:
    return TemplateArgument(ReadDecl(F, Record, Idx));
  case TemplateArgument::Integral: {
    llvm::APSInt Value = ReadAPSInt(Record, Idx);
    QualType T = readType(F, Record, Idx);
    return TemplateArgument(Value, T);
  }
  case TemplateArgument::Template: 
    return TemplateArgument(ReadTemplateName(F, Record, Idx));
  case TemplateArgument::TemplateExpansion: {
    TemplateName Name = ReadTemplateName(F, Record, Idx);
    llvm::Optional<unsigned> NumTemplateExpansions;
    if (unsigned NumExpansions = Record[Idx++])
      NumTemplateExpansions = NumExpansions - 1;
    return TemplateArgument(Name, NumTemplateExpansions);
  }
  case TemplateArgument::Expression:
    return TemplateArgument(ReadExpr(F));
  case TemplateArgument::Pack: {
    unsigned NumArgs = Record[Idx++];
    TemplateArgument *Args = new (Context) TemplateArgument[NumArgs];
    for (unsigned I = 0; I != NumArgs; ++I)
      Args[I] = ReadTemplateArgument(F, Record, Idx);
    return TemplateArgument(Args, NumArgs);
  }
  }

  llvm_unreachable("Unhandled template argument kind!");
}

TemplateParameterList *
ASTReader::ReadTemplateParameterList(ModuleFile &F,
                                     const RecordData &Record, unsigned &Idx) {
  SourceLocation TemplateLoc = ReadSourceLocation(F, Record, Idx);
  SourceLocation LAngleLoc = ReadSourceLocation(F, Record, Idx);
  SourceLocation RAngleLoc = ReadSourceLocation(F, Record, Idx);

  unsigned NumParams = Record[Idx++];
  SmallVector<NamedDecl *, 16> Params;
  Params.reserve(NumParams);
  while (NumParams--)
    Params.push_back(ReadDeclAs<NamedDecl>(F, Record, Idx));

  TemplateParameterList* TemplateParams =
    TemplateParameterList::Create(Context, TemplateLoc, LAngleLoc,
                                  Params.data(), Params.size(), RAngleLoc);
  return TemplateParams;
}

void
ASTReader::
ReadTemplateArgumentList(SmallVector<TemplateArgument, 8> &TemplArgs,
                         ModuleFile &F, const RecordData &Record,
                         unsigned &Idx) {
  unsigned NumTemplateArgs = Record[Idx++];
  TemplArgs.reserve(NumTemplateArgs);
  while (NumTemplateArgs--)
    TemplArgs.push_back(ReadTemplateArgument(F, Record, Idx));
}

/// \brief Read a UnresolvedSet structure.
void ASTReader::ReadUnresolvedSet(ModuleFile &F, UnresolvedSetImpl &Set,
                                  const RecordData &Record, unsigned &Idx) {
  unsigned NumDecls = Record[Idx++];
  while (NumDecls--) {
    NamedDecl *D = ReadDeclAs<NamedDecl>(F, Record, Idx);
    AccessSpecifier AS = (AccessSpecifier)Record[Idx++];
    Set.addDecl(D, AS);
  }
}

CXXBaseSpecifier
ASTReader::ReadCXXBaseSpecifier(ModuleFile &F,
                                const RecordData &Record, unsigned &Idx) {
  bool isVirtual = static_cast<bool>(Record[Idx++]);
  bool isBaseOfClass = static_cast<bool>(Record[Idx++]);
  AccessSpecifier AS = static_cast<AccessSpecifier>(Record[Idx++]);
  bool inheritConstructors = static_cast<bool>(Record[Idx++]);
  TypeSourceInfo *TInfo = GetTypeSourceInfo(F, Record, Idx);
  SourceRange Range = ReadSourceRange(F, Record, Idx);
  SourceLocation EllipsisLoc = ReadSourceLocation(F, Record, Idx);
  CXXBaseSpecifier Result(Range, isVirtual, isBaseOfClass, AS, TInfo, 
                          EllipsisLoc);
  Result.setInheritConstructors(inheritConstructors);
  return Result;
}

std::pair<CXXCtorInitializer **, unsigned>
ASTReader::ReadCXXCtorInitializers(ModuleFile &F, const RecordData &Record,
                                   unsigned &Idx) {
  CXXCtorInitializer **CtorInitializers = 0;
  unsigned NumInitializers = Record[Idx++];
  if (NumInitializers) {
    CtorInitializers
        = new (Context) CXXCtorInitializer*[NumInitializers];
    for (unsigned i=0; i != NumInitializers; ++i) {
      TypeSourceInfo *TInfo = 0;
      bool IsBaseVirtual = false;
      FieldDecl *Member = 0;
      IndirectFieldDecl *IndirectMember = 0;

      CtorInitializerType Type = (CtorInitializerType)Record[Idx++];
      switch (Type) {
      case CTOR_INITIALIZER_BASE:
        TInfo = GetTypeSourceInfo(F, Record, Idx);
        IsBaseVirtual = Record[Idx++];
        break;
          
      case CTOR_INITIALIZER_DELEGATING:
        TInfo = GetTypeSourceInfo(F, Record, Idx);
        break;

       case CTOR_INITIALIZER_MEMBER:
        Member = ReadDeclAs<FieldDecl>(F, Record, Idx);
        break;

       case CTOR_INITIALIZER_INDIRECT_MEMBER:
        IndirectMember = ReadDeclAs<IndirectFieldDecl>(F, Record, Idx);
        break;
      }

      SourceLocation MemberOrEllipsisLoc = ReadSourceLocation(F, Record, Idx);
      Expr *Init = ReadExpr(F);
      SourceLocation LParenLoc = ReadSourceLocation(F, Record, Idx);
      SourceLocation RParenLoc = ReadSourceLocation(F, Record, Idx);
      bool IsWritten = Record[Idx++];
      unsigned SourceOrderOrNumArrayIndices;
      SmallVector<VarDecl *, 8> Indices;
      if (IsWritten) {
        SourceOrderOrNumArrayIndices = Record[Idx++];
      } else {
        SourceOrderOrNumArrayIndices = Record[Idx++];
        Indices.reserve(SourceOrderOrNumArrayIndices);
        for (unsigned i=0; i != SourceOrderOrNumArrayIndices; ++i)
          Indices.push_back(ReadDeclAs<VarDecl>(F, Record, Idx));
      }

      CXXCtorInitializer *BOMInit;
      if (Type == CTOR_INITIALIZER_BASE) {
        BOMInit = new (Context) CXXCtorInitializer(Context, TInfo, IsBaseVirtual,
                                             LParenLoc, Init, RParenLoc,
                                             MemberOrEllipsisLoc);
      } else if (Type == CTOR_INITIALIZER_DELEGATING) {
        BOMInit = new (Context) CXXCtorInitializer(Context, TInfo, LParenLoc,
                                                   Init, RParenLoc);
      } else if (IsWritten) {
        if (Member)
          BOMInit = new (Context) CXXCtorInitializer(Context, Member, MemberOrEllipsisLoc,
                                               LParenLoc, Init, RParenLoc);
        else 
          BOMInit = new (Context) CXXCtorInitializer(Context, IndirectMember,
                                               MemberOrEllipsisLoc, LParenLoc,
                                               Init, RParenLoc);
      } else {
        BOMInit = CXXCtorInitializer::Create(Context, Member, MemberOrEllipsisLoc,
                                             LParenLoc, Init, RParenLoc,
                                             Indices.data(), Indices.size());
      }

      if (IsWritten)
        BOMInit->setSourceOrder(SourceOrderOrNumArrayIndices);
      CtorInitializers[i] = BOMInit;
    }
  }

  return std::make_pair(CtorInitializers, NumInitializers);
}

NestedNameSpecifier *
ASTReader::ReadNestedNameSpecifier(ModuleFile &F,
                                   const RecordData &Record, unsigned &Idx) {
  unsigned N = Record[Idx++];
  NestedNameSpecifier *NNS = 0, *Prev = 0;
  for (unsigned I = 0; I != N; ++I) {
    NestedNameSpecifier::SpecifierKind Kind
      = (NestedNameSpecifier::SpecifierKind)Record[Idx++];
    switch (Kind) {
    case NestedNameSpecifier::Identifier: {
      IdentifierInfo *II = GetIdentifierInfo(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(Context, Prev, II);
      break;
    }

    case NestedNameSpecifier::Namespace: {
      NamespaceDecl *NS = ReadDeclAs<NamespaceDecl>(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(Context, Prev, NS);
      break;
    }

    case NestedNameSpecifier::NamespaceAlias: {
      NamespaceAliasDecl *Alias =ReadDeclAs<NamespaceAliasDecl>(F, Record, Idx);
      NNS = NestedNameSpecifier::Create(Context, Prev, Alias);
      break;
    }

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      const Type *T = readType(F, Record, Idx).getTypePtrOrNull();
      if (!T)
        return 0;
      
      bool Template = Record[Idx++];
      NNS = NestedNameSpecifier::Create(Context, Prev, Template, T);
      break;
    }

    case NestedNameSpecifier::Global: {
      NNS = NestedNameSpecifier::GlobalSpecifier(Context);
      // No associated value, and there can't be a prefix.
      break;
    }
    }
    Prev = NNS;
  }
  return NNS;
}

NestedNameSpecifierLoc
ASTReader::ReadNestedNameSpecifierLoc(ModuleFile &F, const RecordData &Record, 
                                      unsigned &Idx) {
  unsigned N = Record[Idx++];
  NestedNameSpecifierLocBuilder Builder;
  for (unsigned I = 0; I != N; ++I) {
    NestedNameSpecifier::SpecifierKind Kind
      = (NestedNameSpecifier::SpecifierKind)Record[Idx++];
    switch (Kind) {
    case NestedNameSpecifier::Identifier: {
      IdentifierInfo *II = GetIdentifierInfo(F, Record, Idx);      
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(Context, II, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::Namespace: {
      NamespaceDecl *NS = ReadDeclAs<NamespaceDecl>(F, Record, Idx);
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(Context, NS, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::NamespaceAlias: {
      NamespaceAliasDecl *Alias =ReadDeclAs<NamespaceAliasDecl>(F, Record, Idx);
      SourceRange Range = ReadSourceRange(F, Record, Idx);
      Builder.Extend(Context, Alias, Range.getBegin(), Range.getEnd());
      break;
    }

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate: {
      bool Template = Record[Idx++];
      TypeSourceInfo *T = GetTypeSourceInfo(F, Record, Idx);
      if (!T)
        return NestedNameSpecifierLoc();
      SourceLocation ColonColonLoc = ReadSourceLocation(F, Record, Idx);

      // FIXME: 'template' keyword location not saved anywhere, so we fake it.
      Builder.Extend(Context, 
                     Template? T->getTypeLoc().getBeginLoc() : SourceLocation(),
                     T->getTypeLoc(), ColonColonLoc);
      break;
    }

    case NestedNameSpecifier::Global: {
      SourceLocation ColonColonLoc = ReadSourceLocation(F, Record, Idx);
      Builder.MakeGlobal(Context, ColonColonLoc);
      break;
    }
    }
  }
  
  return Builder.getWithLocInContext(Context);
}

SourceRange
ASTReader::ReadSourceRange(ModuleFile &F, const RecordData &Record,
                           unsigned &Idx) {
  SourceLocation beg = ReadSourceLocation(F, Record, Idx);
  SourceLocation end = ReadSourceLocation(F, Record, Idx);
  return SourceRange(beg, end);
}

/// \brief Read an integral value
llvm::APInt ASTReader::ReadAPInt(const RecordData &Record, unsigned &Idx) {
  unsigned BitWidth = Record[Idx++];
  unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
  llvm::APInt Result(BitWidth, NumWords, &Record[Idx]);
  Idx += NumWords;
  return Result;
}

/// \brief Read a signed integral value
llvm::APSInt ASTReader::ReadAPSInt(const RecordData &Record, unsigned &Idx) {
  bool isUnsigned = Record[Idx++];
  return llvm::APSInt(ReadAPInt(Record, Idx), isUnsigned);
}

/// \brief Read a floating-point value
llvm::APFloat ASTReader::ReadAPFloat(const RecordData &Record, unsigned &Idx) {
  return llvm::APFloat(ReadAPInt(Record, Idx));
}

// \brief Read a string
std::string ASTReader::ReadString(const RecordData &Record, unsigned &Idx) {
  unsigned Len = Record[Idx++];
  std::string Result(Record.data() + Idx, Record.data() + Idx + Len);
  Idx += Len;
  return Result;
}

VersionTuple ASTReader::ReadVersionTuple(const RecordData &Record, 
                                         unsigned &Idx) {
  unsigned Major = Record[Idx++];
  unsigned Minor = Record[Idx++];
  unsigned Subminor = Record[Idx++];
  if (Minor == 0)
    return VersionTuple(Major);
  if (Subminor == 0)
    return VersionTuple(Major, Minor - 1);
  return VersionTuple(Major, Minor - 1, Subminor - 1);
}

CXXTemporary *ASTReader::ReadCXXTemporary(ModuleFile &F, 
                                          const RecordData &Record,
                                          unsigned &Idx) {
  CXXDestructorDecl *Decl = ReadDeclAs<CXXDestructorDecl>(F, Record, Idx);
  return CXXTemporary::Create(Context, Decl);
}

DiagnosticBuilder ASTReader::Diag(unsigned DiagID) {
  return Diag(SourceLocation(), DiagID);
}

DiagnosticBuilder ASTReader::Diag(SourceLocation Loc, unsigned DiagID) {
  return Diags.Report(Loc, DiagID);
}

/// \brief Retrieve the identifier table associated with the
/// preprocessor.
IdentifierTable &ASTReader::getIdentifierTable() {
  return PP.getIdentifierTable();
}

/// \brief Record that the given ID maps to the given switch-case
/// statement.
void ASTReader::RecordSwitchCaseID(SwitchCase *SC, unsigned ID) {
  assert(SwitchCaseStmts[ID] == 0 && "Already have a SwitchCase with this ID");
  SwitchCaseStmts[ID] = SC;
}

/// \brief Retrieve the switch-case statement with the given ID.
SwitchCase *ASTReader::getSwitchCaseWithID(unsigned ID) {
  assert(SwitchCaseStmts[ID] != 0 && "No SwitchCase with this ID");
  return SwitchCaseStmts[ID];
}

void ASTReader::ClearSwitchCaseIDs() {
  SwitchCaseStmts.clear();
}

void ASTReader::finishPendingActions() {
  while (!PendingIdentifierInfos.empty() || !PendingDeclChains.empty()) {
    // If any identifiers with corresponding top-level declarations have
    // been loaded, load those declarations now.
    while (!PendingIdentifierInfos.empty()) {
      SetGloballyVisibleDecls(PendingIdentifierInfos.front().II,
                              PendingIdentifierInfos.front().DeclIDs, true);
      PendingIdentifierInfos.pop_front();
    }
  
    // Load pending declaration chains.
    for (unsigned I = 0; I != PendingDeclChains.size(); ++I) {
      loadPendingDeclChain(PendingDeclChains[I]);
      PendingDeclChainsKnown.erase(PendingDeclChains[I]);
    }
    PendingDeclChains.clear();
  }
  
  // If we deserialized any C++ or Objective-C class definitions, any
  // Objective-C protocol definitions, or any redeclarable templates, make sure
  // that all redeclarations point to the definitions. Note that this can only 
  // happen now, after the redeclaration chains have been fully wired.
  for (llvm::SmallPtrSet<Decl *, 4>::iterator D = PendingDefinitions.begin(),
                                           DEnd = PendingDefinitions.end();
       D != DEnd; ++D) {
    if (TagDecl *TD = dyn_cast<TagDecl>(*D)) {
      if (const TagType *TagT = dyn_cast<TagType>(TD->TypeForDecl)) {
        // Make sure that the TagType points at the definition.
        const_cast<TagType*>(TagT)->decl = TD;
      }
      
      if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(*D)) {
        for (CXXRecordDecl::redecl_iterator R = RD->redecls_begin(),
                                         REnd = RD->redecls_end();
             R != REnd; ++R)
          cast<CXXRecordDecl>(*R)->DefinitionData = RD->DefinitionData;
        
      }

      continue;
    }
    
    if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(*D)) {
      // Make sure that the ObjCInterfaceType points at the definition.
      const_cast<ObjCInterfaceType *>(cast<ObjCInterfaceType>(ID->TypeForDecl))
        ->Decl = ID;
      
      for (ObjCInterfaceDecl::redecl_iterator R = ID->redecls_begin(),
                                           REnd = ID->redecls_end();
           R != REnd; ++R)
        R->Data = ID->Data;
      
      continue;
    }
    
    if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(*D)) {
      for (ObjCProtocolDecl::redecl_iterator R = PD->redecls_begin(),
                                          REnd = PD->redecls_end();
           R != REnd; ++R)
        R->Data = PD->Data;
      
      continue;
    }
    
    RedeclarableTemplateDecl *RTD
      = cast<RedeclarableTemplateDecl>(*D)->getCanonicalDecl();
    for (RedeclarableTemplateDecl::redecl_iterator R = RTD->redecls_begin(),
                                                REnd = RTD->redecls_end();
         R != REnd; ++R)
      R->Common = RTD->Common;    
  }
  PendingDefinitions.clear();
}

void ASTReader::FinishedDeserializing() {
  assert(NumCurrentElementsDeserializing &&
         "FinishedDeserializing not paired with StartedDeserializing");
  if (NumCurrentElementsDeserializing == 1) {
    // We decrease NumCurrentElementsDeserializing only after pending actions
    // are finished, to avoid recursively re-calling finishPendingActions().
    finishPendingActions();
  }
  --NumCurrentElementsDeserializing;

  if (NumCurrentElementsDeserializing == 0 &&
      Consumer && !PassingDeclsToConsumer) {
    // Guard variable to avoid recursively redoing the process of passing
    // decls to consumer.
    SaveAndRestore<bool> GuardPassingDeclsToConsumer(PassingDeclsToConsumer,
                                                     true);

    while (!InterestingDecls.empty()) {
      // We are not in recursive loading, so it's safe to pass the "interesting"
      // decls to the consumer.
      Decl *D = InterestingDecls.front();
      InterestingDecls.pop_front();
      PassInterestingDeclToConsumer(D);
    }
  }
}

ASTReader::ASTReader(Preprocessor &PP, ASTContext &Context,
                     StringRef isysroot, bool DisableValidation,
                     bool DisableStatCache, bool AllowASTWithCompilerErrors)
  : Listener(new PCHValidator(PP, *this)), DeserializationListener(0),
    SourceMgr(PP.getSourceManager()), FileMgr(PP.getFileManager()),
    Diags(PP.getDiagnostics()), SemaObj(0), PP(PP), Context(Context),
    Consumer(0), ModuleMgr(FileMgr.getFileSystemOptions()),
    RelocatablePCH(false), isysroot(isysroot),
    DisableValidation(DisableValidation),
    DisableStatCache(DisableStatCache),
    AllowASTWithCompilerErrors(AllowASTWithCompilerErrors), 
    CurrentGeneration(0), NumStatHits(0), NumStatMisses(0), 
    NumSLocEntriesRead(0), TotalNumSLocEntries(0), 
    NumStatementsRead(0), TotalNumStatements(0), NumMacrosRead(0), 
    TotalNumMacros(0), NumSelectorsRead(0), NumMethodPoolEntriesRead(0), 
    NumMethodPoolMisses(0), TotalNumMethodPoolEntries(0), 
    NumLexicalDeclContextsRead(0), TotalLexicalDeclContexts(0), 
    NumVisibleDeclContextsRead(0), TotalVisibleDeclContexts(0),
    TotalModulesSizeInBits(0), NumCurrentElementsDeserializing(0),
    PassingDeclsToConsumer(false),
    NumCXXBaseSpecifiersLoaded(0)
{
  SourceMgr.setExternalSLocEntrySource(this);
}

ASTReader::~ASTReader() {
  for (DeclContextVisibleUpdatesPending::iterator
           I = PendingVisibleUpdates.begin(),
           E = PendingVisibleUpdates.end();
       I != E; ++I) {
    for (DeclContextVisibleUpdates::iterator J = I->second.begin(),
                                             F = I->second.end();
         J != F; ++J)
      delete J->first;
  }
}
