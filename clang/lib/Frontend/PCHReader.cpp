//===--- PCHReader.cpp - Precompiled Headers Reader -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHReader class, which reads a precompiled header.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "../Sema/Sema.h" // FIXME: move Sema headers elsewhere
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <sys/stat.h>
using namespace clang;

//===----------------------------------------------------------------------===//
// PCH reader validator implementation
//===----------------------------------------------------------------------===//

PCHReaderListener::~PCHReaderListener() {}

bool
PCHValidator::ReadLanguageOptions(const LangOptions &LangOpts) {
  const LangOptions &PPLangOpts = PP.getLangOptions();
#define PARSE_LANGOPT_BENIGN(Option)
#define PARSE_LANGOPT_IMPORTANT(Option, DiagID)                    \
  if (PPLangOpts.Option != LangOpts.Option) {                      \
    Reader.Diag(DiagID) << LangOpts.Option << PPLangOpts.Option;   \
    return true;                                                   \
  }

  PARSE_LANGOPT_BENIGN(Trigraphs);
  PARSE_LANGOPT_BENIGN(BCPLComment);
  PARSE_LANGOPT_BENIGN(DollarIdents);
  PARSE_LANGOPT_BENIGN(AsmPreprocessor);
  PARSE_LANGOPT_IMPORTANT(GNUMode, diag::warn_pch_gnu_extensions);
  PARSE_LANGOPT_BENIGN(ImplicitInt);
  PARSE_LANGOPT_BENIGN(Digraphs);
  PARSE_LANGOPT_BENIGN(HexFloats);
  PARSE_LANGOPT_IMPORTANT(C99, diag::warn_pch_c99);
  PARSE_LANGOPT_IMPORTANT(Microsoft, diag::warn_pch_microsoft_extensions);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus, diag::warn_pch_cplusplus);
  PARSE_LANGOPT_IMPORTANT(CPlusPlus0x, diag::warn_pch_cplusplus0x);
  PARSE_LANGOPT_BENIGN(CXXOperatorName);
  PARSE_LANGOPT_IMPORTANT(ObjC1, diag::warn_pch_objective_c);
  PARSE_LANGOPT_IMPORTANT(ObjC2, diag::warn_pch_objective_c2);
  PARSE_LANGOPT_IMPORTANT(ObjCNonFragileABI, diag::warn_pch_nonfragile_abi);
  PARSE_LANGOPT_BENIGN(PascalStrings);
  PARSE_LANGOPT_BENIGN(WritableStrings);
  PARSE_LANGOPT_IMPORTANT(LaxVectorConversions, 
                          diag::warn_pch_lax_vector_conversions);
  PARSE_LANGOPT_IMPORTANT(AltiVec, diag::warn_pch_altivec);
  PARSE_LANGOPT_IMPORTANT(Exceptions, diag::warn_pch_exceptions);
  PARSE_LANGOPT_IMPORTANT(NeXTRuntime, diag::warn_pch_objc_runtime);
  PARSE_LANGOPT_IMPORTANT(Freestanding, diag::warn_pch_freestanding);
  PARSE_LANGOPT_IMPORTANT(NoBuiltin, diag::warn_pch_builtins);
  PARSE_LANGOPT_IMPORTANT(ThreadsafeStatics, 
                          diag::warn_pch_thread_safe_statics);
  PARSE_LANGOPT_IMPORTANT(Blocks, diag::warn_pch_blocks);
  PARSE_LANGOPT_BENIGN(EmitAllDecls);
  PARSE_LANGOPT_IMPORTANT(MathErrno, diag::warn_pch_math_errno);
  PARSE_LANGOPT_IMPORTANT(OverflowChecking, diag::warn_pch_overflow_checking);
  PARSE_LANGOPT_IMPORTANT(HeinousExtensions, 
                          diag::warn_pch_heinous_extensions);
  // FIXME: Most of the options below are benign if the macro wasn't
  // used. Unfortunately, this means that a PCH compiled without
  // optimization can't be used with optimization turned on, even
  // though the only thing that changes is whether __OPTIMIZE__ was
  // defined... but if __OPTIMIZE__ never showed up in the header, it
  // doesn't matter. We could consider making this some special kind
  // of check.
  PARSE_LANGOPT_IMPORTANT(Optimize, diag::warn_pch_optimize);
  PARSE_LANGOPT_IMPORTANT(OptimizeSize, diag::warn_pch_optimize_size);
  PARSE_LANGOPT_IMPORTANT(Static, diag::warn_pch_static);
  PARSE_LANGOPT_IMPORTANT(PICLevel, diag::warn_pch_pic_level);
  PARSE_LANGOPT_IMPORTANT(GNUInline, diag::warn_pch_gnu_inline);
  PARSE_LANGOPT_IMPORTANT(NoInline, diag::warn_pch_no_inline);
  PARSE_LANGOPT_IMPORTANT(AccessControl, diag::warn_pch_access_control);
  PARSE_LANGOPT_IMPORTANT(CharIsSigned, diag::warn_pch_char_signed);
  if ((PPLangOpts.getGCMode() != 0) != (LangOpts.getGCMode() != 0)) {
    Reader.Diag(diag::warn_pch_gc_mode) 
      << LangOpts.getGCMode() << PPLangOpts.getGCMode();
    return true;
  }
  PARSE_LANGOPT_BENIGN(getVisibilityMode());
  PARSE_LANGOPT_BENIGN(InstantiationDepth);
  PARSE_LANGOPT_IMPORTANT(OpenCL, diag::warn_pch_opencl);
#undef PARSE_LANGOPT_IRRELEVANT
#undef PARSE_LANGOPT_BENIGN

  return false;
}

bool PCHValidator::ReadTargetTriple(const std::string &Triple) {
  if (Triple != PP.getTargetInfo().getTargetTriple()) {
    Reader.Diag(diag::warn_pch_target_triple)
      << Triple << PP.getTargetInfo().getTargetTriple();
    return true;
  }
  return false;
}

/// \brief Split the given string into a vector of lines, eliminating
/// any empty lines in the process.
///
/// \param Str the string to split.
/// \param Len the length of Str.
/// \param KeepEmptyLines true if empty lines should be included
/// \returns a vector of lines, with the line endings removed
static std::vector<std::string> splitLines(const char *Str, unsigned Len,
                                           bool KeepEmptyLines = false) {
  std::vector<std::string> Lines;
  for (unsigned LineStart = 0; LineStart < Len; ++LineStart) {
    unsigned LineEnd = LineStart;
    while (LineEnd < Len && Str[LineEnd] != '\n')
      ++LineEnd;
    if (LineStart != LineEnd || KeepEmptyLines)
      Lines.push_back(std::string(&Str[LineStart], &Str[LineEnd]));
    LineStart = LineEnd;
  }
  return Lines;
}

/// \brief Determine whether the string Haystack starts with the
/// substring Needle.
static bool startsWith(const std::string &Haystack, const char *Needle) {
  for (unsigned I = 0, N = Haystack.size(); Needle[I] != 0; ++I) {
    if (I == N)
      return false;
    if (Haystack[I] != Needle[I])
      return false;
  }

  return true;
}

/// \brief Determine whether the string Haystack starts with the
/// substring Needle.
static inline bool startsWith(const std::string &Haystack,
                              const std::string &Needle) {
  return startsWith(Haystack, Needle.c_str());
}

bool PCHValidator::ReadPredefinesBuffer(const char *PCHPredef, 
                                        unsigned PCHPredefLen,
                                        FileID PCHBufferID,
                                        std::string &SuggestedPredefines) {
  const char *Predef = PP.getPredefines().c_str();
  unsigned PredefLen = PP.getPredefines().size();

  // If the two predefines buffers compare equal, we're done!
  if (PredefLen == PCHPredefLen && 
      strncmp(Predef, PCHPredef, PCHPredefLen) == 0)
    return false;

  SourceManager &SourceMgr = PP.getSourceManager();
  
  // The predefines buffers are different. Determine what the
  // differences are, and whether they require us to reject the PCH
  // file.
  std::vector<std::string> CmdLineLines = splitLines(Predef, PredefLen);
  std::vector<std::string> PCHLines = splitLines(PCHPredef, PCHPredefLen);

  // Sort both sets of predefined buffer lines, since 
  std::sort(CmdLineLines.begin(), CmdLineLines.end());
  std::sort(PCHLines.begin(), PCHLines.end());

  // Determine which predefines that where used to build the PCH file
  // are missing from the command line.
  std::vector<std::string> MissingPredefines;
  std::set_difference(PCHLines.begin(), PCHLines.end(),
                      CmdLineLines.begin(), CmdLineLines.end(),
                      std::back_inserter(MissingPredefines));

  bool MissingDefines = false;
  bool ConflictingDefines = false;
  for (unsigned I = 0, N = MissingPredefines.size(); I != N; ++I) {
    const std::string &Missing = MissingPredefines[I];
    if (!startsWith(Missing, "#define ") != 0) {
      Reader.Diag(diag::warn_pch_compiler_options_mismatch);
      return true;
    }
    
    // This is a macro definition. Determine the name of the macro
    // we're defining.
    std::string::size_type StartOfMacroName = strlen("#define ");
    std::string::size_type EndOfMacroName 
      = Missing.find_first_of("( \n\r", StartOfMacroName);
    assert(EndOfMacroName != std::string::npos &&
           "Couldn't find the end of the macro name");
    std::string MacroName = Missing.substr(StartOfMacroName,
                                           EndOfMacroName - StartOfMacroName);

    // Determine whether this macro was given a different definition
    // on the command line.
    std::string MacroDefStart = "#define " + MacroName;
    std::string::size_type MacroDefLen = MacroDefStart.size();
    std::vector<std::string>::iterator ConflictPos
      = std::lower_bound(CmdLineLines.begin(), CmdLineLines.end(),
                         MacroDefStart);
    for (; ConflictPos != CmdLineLines.end(); ++ConflictPos) {
      if (!startsWith(*ConflictPos, MacroDefStart)) {
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
      const char *MissingDef = strstr(PCHPredef, Missing.c_str());
      unsigned Offset = MissingDef - PCHPredef;
      SourceLocation PCHMissingLoc
        = SourceMgr.getLocForStartOfFile(PCHBufferID)
            .getFileLocWithOffset(Offset);
      Reader.Diag(PCHMissingLoc, diag::note_pch_macro_defined_as)
        << MacroName;

      ConflictingDefines = true;
      continue;
    }
    
    // If the macro doesn't conflict, then we'll just pick up the
    // macro definition from the PCH file. Warn the user that they
    // made a mistake.
    if (ConflictingDefines)
      continue; // Don't complain if there are already conflicting defs
    
    if (!MissingDefines) {
      Reader.Diag(diag::warn_cmdline_missing_macro_defs);
      MissingDefines = true;
    }

    // Show the definition of this macro within the PCH file.
    const char *MissingDef = strstr(PCHPredef, Missing.c_str());
    unsigned Offset = MissingDef - PCHPredef;
    SourceLocation PCHMissingLoc
      = SourceMgr.getLocForStartOfFile(PCHBufferID)
      .getFileLocWithOffset(Offset);
    Reader.Diag(PCHMissingLoc, diag::note_using_macro_def_from_pch);
  }
  
  if (ConflictingDefines)
    return true;
  
  // Determine what predefines were introduced based on command-line
  // parameters that were not present when building the PCH
  // file. Extra #defines are okay, so long as the identifiers being
  // defined were not used within the precompiled header.
  std::vector<std::string> ExtraPredefines;
  std::set_difference(CmdLineLines.begin(), CmdLineLines.end(),
                      PCHLines.begin(), PCHLines.end(),
                      std::back_inserter(ExtraPredefines));  
  for (unsigned I = 0, N = ExtraPredefines.size(); I != N; ++I) {
    const std::string &Extra = ExtraPredefines[I];
    if (!startsWith(Extra, "#define ") != 0) {
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
    std::string MacroName = Extra.substr(StartOfMacroName,
                                         EndOfMacroName - StartOfMacroName);

    // Check whether this name was used somewhere in the PCH file. If
    // so, defining it as a macro could change behavior, so we reject
    // the PCH file.
    if (IdentifierInfo *II = Reader.get(MacroName.c_str(),
                                 MacroName.c_str() + MacroName.size())) {
      Reader.Diag(diag::warn_macro_name_used_in_pch)
        << II;
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

void PCHValidator::ReadHeaderFileInfo(const HeaderFileInfo &HFI) {
  PP.getHeaderSearchInfo().setHeaderFileInfoForUID(HFI, NumHeaderInfos++);
}

void PCHValidator::ReadCounter(unsigned Value) {
  PP.setCounterValue(Value);
}



//===----------------------------------------------------------------------===//
// PCH reader implementation
//===----------------------------------------------------------------------===//

PCHReader::PCHReader(Preprocessor &PP, ASTContext *Context, 
                     const char *isysroot) 
  : Listener(new PCHValidator(PP, *this)), SourceMgr(PP.getSourceManager()),
    FileMgr(PP.getFileManager()), Diags(PP.getDiagnostics()),
    SemaObj(0), PP(&PP), Context(Context), Consumer(0),
    IdentifierTableData(0), IdentifierLookupTable(0),
    IdentifierOffsets(0),
    MethodPoolLookupTable(0), MethodPoolLookupTableData(0),
    TotalSelectorsInMethodPool(0), SelectorOffsets(0),
    TotalNumSelectors(0), Comments(0), NumComments(0), isysroot(isysroot),
    NumStatHits(0), NumStatMisses(0), 
    NumSLocEntriesRead(0), NumStatementsRead(0), 
    NumMacrosRead(0), NumMethodPoolSelectorsRead(0), NumMethodPoolMisses(0),
    NumLexicalDeclContextsRead(0), NumVisibleDeclContextsRead(0),
    CurrentlyLoadingTypeOrDecl(0) { 
  RelocatablePCH = false;
}

PCHReader::PCHReader(SourceManager &SourceMgr, FileManager &FileMgr,
                     Diagnostic &Diags, const char *isysroot) 
  : SourceMgr(SourceMgr), FileMgr(FileMgr), Diags(Diags),
    SemaObj(0), PP(0), Context(0), Consumer(0),
    IdentifierTableData(0), IdentifierLookupTable(0),
    IdentifierOffsets(0),
    MethodPoolLookupTable(0), MethodPoolLookupTableData(0),
    TotalSelectorsInMethodPool(0), SelectorOffsets(0),
    TotalNumSelectors(0), Comments(0), NumComments(0), isysroot(isysroot),
    NumStatHits(0), NumStatMisses(0), 
    NumSLocEntriesRead(0), NumStatementsRead(0), 
    NumMacrosRead(0), NumMethodPoolSelectorsRead(0), NumMethodPoolMisses(0),
    NumLexicalDeclContextsRead(0), NumVisibleDeclContextsRead(0),
    CurrentlyLoadingTypeOrDecl(0) { 
  RelocatablePCH = false;
}

PCHReader::~PCHReader() {}

Expr *PCHReader::ReadDeclExpr() {
  return dyn_cast_or_null<Expr>(ReadStmt(DeclsCursor));
}

Expr *PCHReader::ReadTypeExpr() {
  return dyn_cast_or_null<Expr>(ReadStmt(Stream));
}


namespace {
class VISIBILITY_HIDDEN PCHMethodPoolLookupTrait {
  PCHReader &Reader;

public:
  typedef std::pair<ObjCMethodList, ObjCMethodList> data_type;

  typedef Selector external_key_type;
  typedef external_key_type internal_key_type;

  explicit PCHMethodPoolLookupTrait(PCHReader &Reader) : Reader(Reader) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a == b;
  }
  
  static unsigned ComputeHash(Selector Sel) {
    unsigned N = Sel.getNumArgs();
    if (N == 0)
      ++N;
    unsigned R = 5381;
    for (unsigned I = 0; I != N; ++I)
      if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
        R = clang::BernsteinHashPartial(II->getName(), II->getLength(), R);
    return R;
  }
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned KeyLen = ReadUnalignedLE16(d);
    unsigned DataLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }
    
  internal_key_type ReadKey(const unsigned char* d, unsigned) {
    using namespace clang::io;
    SelectorTable &SelTable = Reader.getContext()->Selectors;
    unsigned N = ReadUnalignedLE16(d);
    IdentifierInfo *FirstII 
      = Reader.DecodeIdentifierInfo(ReadUnalignedLE32(d));
    if (N == 0)
      return SelTable.getNullarySelector(FirstII);
    else if (N == 1)
      return SelTable.getUnarySelector(FirstII);

    llvm::SmallVector<IdentifierInfo *, 16> Args;
    Args.push_back(FirstII);
    for (unsigned I = 1; I != N; ++I)
      Args.push_back(Reader.DecodeIdentifierInfo(ReadUnalignedLE32(d)));

    return SelTable.getSelector(N, Args.data());
  }
    
  data_type ReadData(Selector, const unsigned char* d, unsigned DataLen) {
    using namespace clang::io;
    unsigned NumInstanceMethods = ReadUnalignedLE16(d);
    unsigned NumFactoryMethods = ReadUnalignedLE16(d);

    data_type Result;

    // Load instance methods
    ObjCMethodList *Prev = 0;
    for (unsigned I = 0; I != NumInstanceMethods; ++I) {
      ObjCMethodDecl *Method 
        = cast<ObjCMethodDecl>(Reader.GetDecl(ReadUnalignedLE32(d)));
      if (!Result.first.Method) {
        // This is the first method, which is the easy case.
        Result.first.Method = Method;
        Prev = &Result.first;
        continue;
      }

      Prev->Next = new ObjCMethodList(Method, 0);
      Prev = Prev->Next;
    }

    // Load factory methods
    Prev = 0;
    for (unsigned I = 0; I != NumFactoryMethods; ++I) {
      ObjCMethodDecl *Method 
        = cast<ObjCMethodDecl>(Reader.GetDecl(ReadUnalignedLE32(d)));
      if (!Result.second.Method) {
        // This is the first method, which is the easy case.
        Result.second.Method = Method;
        Prev = &Result.second;
        continue;
      }

      Prev->Next = new ObjCMethodList(Method, 0);
      Prev = Prev->Next;
    }

    return Result;
  }
};
  
} // end anonymous namespace  

/// \brief The on-disk hash table used for the global method pool.
typedef OnDiskChainedHashTable<PCHMethodPoolLookupTrait> 
  PCHMethodPoolLookupTable;

namespace {
class VISIBILITY_HIDDEN PCHIdentifierLookupTrait {
  PCHReader &Reader;

  // If we know the IdentifierInfo in advance, it is here and we will
  // not build a new one. Used when deserializing information about an
  // identifier that was constructed before the PCH file was read.
  IdentifierInfo *KnownII;

public:
  typedef IdentifierInfo * data_type;

  typedef const std::pair<const char*, unsigned> external_key_type;

  typedef external_key_type internal_key_type;

  explicit PCHIdentifierLookupTrait(PCHReader &Reader, IdentifierInfo *II = 0) 
    : Reader(Reader), KnownII(II) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return (a.second == b.second) ? memcmp(a.first, b.first, a.second) == 0
                                  : false;
  }
  
  static unsigned ComputeHash(const internal_key_type& a) {
    return BernsteinHash(a.first, a.second);
  }
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d) {
    using namespace clang::io;
    unsigned DataLen = ReadUnalignedLE16(d);
    unsigned KeyLen = ReadUnalignedLE16(d);
    return std::make_pair(KeyLen, DataLen);
  }
    
  static std::pair<const char*, unsigned>
  ReadKey(const unsigned char* d, unsigned n) {
    assert(n >= 2 && d[n-1] == '\0');
    return std::make_pair((const char*) d, n-1);
  }
    
  IdentifierInfo *ReadData(const internal_key_type& k, 
                           const unsigned char* d,
                           unsigned DataLen) {
    using namespace clang::io;
    pch::IdentID ID = ReadUnalignedLE32(d);
    bool IsInteresting = ID & 0x01;

    // Wipe out the "is interesting" bit.
    ID = ID >> 1;

    if (!IsInteresting) {
      // For unintersting identifiers, just build the IdentifierInfo
      // and associate it with the persistent ID.
      IdentifierInfo *II = KnownII;
      if (!II)
        II = &Reader.getIdentifierTable().CreateIdentifierInfo(
                                                 k.first, k.first + k.second);
      Reader.SetIdentifierInfo(ID, II);
      return II;
    }

    unsigned Bits = ReadUnalignedLE16(d);
    bool CPlusPlusOperatorKeyword = Bits & 0x01;
    Bits >>= 1;
    bool Poisoned = Bits & 0x01;
    Bits >>= 1;
    bool ExtensionToken = Bits & 0x01;
    Bits >>= 1;
    bool hasMacroDefinition = Bits & 0x01;
    Bits >>= 1;
    unsigned ObjCOrBuiltinID = Bits & 0x3FF;
    Bits >>= 10;
    
    assert(Bits == 0 && "Extra bits in the identifier?");
    DataLen -= 6;

    // Build the IdentifierInfo itself and link the identifier ID with
    // the new IdentifierInfo.
    IdentifierInfo *II = KnownII;
    if (!II)
      II = &Reader.getIdentifierTable().CreateIdentifierInfo(
                                                 k.first, k.first + k.second);
    Reader.SetIdentifierInfo(ID, II);

    // Set or check the various bits in the IdentifierInfo structure.
    // FIXME: Load token IDs lazily, too?
    II->setObjCOrBuiltinID(ObjCOrBuiltinID);
    assert(II->isExtensionToken() == ExtensionToken && 
           "Incorrect extension token flag");
    (void)ExtensionToken;
    II->setIsPoisoned(Poisoned);
    assert(II->isCPlusPlusOperatorKeyword() == CPlusPlusOperatorKeyword &&
           "Incorrect C++ operator keyword flag");
    (void)CPlusPlusOperatorKeyword;

    // If this identifier is a macro, deserialize the macro
    // definition.
    if (hasMacroDefinition) {
      uint32_t Offset = ReadUnalignedLE32(d);
      Reader.ReadMacroRecord(Offset);
      DataLen -= 4;
    }

    // Read all of the declarations visible at global scope with this
    // name.
    if (Reader.getContext() == 0) return II;
    if (DataLen > 0) {
      llvm::SmallVector<uint32_t, 4> DeclIDs;
      for (; DataLen > 0; DataLen -= 4)
        DeclIDs.push_back(ReadUnalignedLE32(d));
      Reader.SetGloballyVisibleDecls(II, DeclIDs);
    }
    
    return II;
  }
};
  
} // end anonymous namespace  

/// \brief The on-disk hash table used to contain information about
/// all of the identifiers in the program.
typedef OnDiskChainedHashTable<PCHIdentifierLookupTrait> 
  PCHIdentifierLookupTable;

bool PCHReader::Error(const char *Msg) {
  unsigned DiagID = Diags.getCustomDiagID(Diagnostic::Fatal, Msg);
  Diag(DiagID);
  return true;
}

/// \brief Check the contents of the predefines buffer against the
/// contents of the predefines buffer used to build the PCH file.
///
/// The contents of the two predefines buffers should be the same. If
/// not, then some command-line option changed the preprocessor state
/// and we must reject the PCH file.
///
/// \param PCHPredef The start of the predefines buffer in the PCH
/// file.
///
/// \param PCHPredefLen The length of the predefines buffer in the PCH
/// file.
///
/// \param PCHBufferID The FileID for the PCH predefines buffer.
///
/// \returns true if there was a mismatch (in which case the PCH file
/// should be ignored), or false otherwise.
bool PCHReader::CheckPredefinesBuffer(const char *PCHPredef, 
                                      unsigned PCHPredefLen,
                                      FileID PCHBufferID) {
  if (Listener)
    return Listener->ReadPredefinesBuffer(PCHPredef, PCHPredefLen, PCHBufferID,
                                          SuggestedPredefines);
  return false;
}

//===----------------------------------------------------------------------===//
// Source Manager Deserialization
//===----------------------------------------------------------------------===//

/// \brief Read the line table in the source manager block.
/// \returns true if ther was an error.
bool PCHReader::ParseLineTable(llvm::SmallVectorImpl<uint64_t> &Record) {
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
    FileIDs[I] = LineTable.getLineTableFilenameID(Filename.c_str(), 
                                                  Filename.size());
  }

  // Parse the line entries
  std::vector<LineEntry> Entries;
  while (Idx < Record.size()) {
    int FID = FileIDs[Record[Idx++]];

    // Extract the line entries
    unsigned NumEntries = Record[Idx++];
    Entries.clear();
    Entries.reserve(NumEntries);
    for (unsigned I = 0; I != NumEntries; ++I) {
      unsigned FileOffset = Record[Idx++];
      unsigned LineNo = Record[Idx++];
      int FilenameID = Record[Idx++];
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

class VISIBILITY_HIDDEN PCHStatData {
public:
  const bool hasStat;
  const ino_t ino;
  const dev_t dev;
  const mode_t mode;
  const time_t mtime;
  const off_t size;
  
  PCHStatData(ino_t i, dev_t d, mode_t mo, time_t m, off_t s)
  : hasStat(true), ino(i), dev(d), mode(mo), mtime(m), size(s) {}  
  
  PCHStatData()
    : hasStat(false), ino(0), dev(0), mode(0), mtime(0), size(0) {}
};

class VISIBILITY_HIDDEN PCHStatLookupTrait {
 public:
  typedef const char *external_key_type;
  typedef const char *internal_key_type;

  typedef PCHStatData data_type;

  static unsigned ComputeHash(const char *path) {
    return BernsteinHash(path);
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

    if (*d++ == 1)
      return data_type();

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
class VISIBILITY_HIDDEN PCHStatCache : public StatSysCallCache {
  typedef OnDiskChainedHashTable<PCHStatLookupTrait> CacheTy;
  CacheTy *Cache;

  unsigned &NumStatHits, &NumStatMisses;
public:  
  PCHStatCache(const unsigned char *Buckets,
               const unsigned char *Base,
               unsigned &NumStatHits,
               unsigned &NumStatMisses) 
    : Cache(0), NumStatHits(NumStatHits), NumStatMisses(NumStatMisses) {
    Cache = CacheTy::Create(Buckets, Base);
  }

  ~PCHStatCache() { delete Cache; }
  
  int stat(const char *path, struct stat *buf) {
    // Do the lookup for the file's data in the PCH file.
    CacheTy::iterator I = Cache->find(path);

    // If we don't get a hit in the PCH file just forward to 'stat'.
    if (I == Cache->end()) {
      ++NumStatMisses;
      return ::stat(path, buf);
    }
    
    ++NumStatHits;
    PCHStatData Data = *I;
    
    if (!Data.hasStat)
      return 1;

    buf->st_ino = Data.ino;
    buf->st_dev = Data.dev;
    buf->st_mtime = Data.mtime;
    buf->st_mode = Data.mode;
    buf->st_size = Data.size;
    return 0;
  }
};
} // end anonymous namespace


/// \brief Read the source manager block
PCHReader::PCHReadResult PCHReader::ReadSourceManagerBlock() {
  using namespace SrcMgr;

  // Set the source-location entry cursor to the current position in
  // the stream. This cursor will be used to read the contents of the
  // source manager block initially, and then lazily read
  // source-location entries as needed.
  SLocEntryCursor = Stream;

  // The stream itself is going to skip over the source manager block.
  if (Stream.SkipBlock()) {
    Error("malformed block record in PCH file");
    return Failure;
  }

  // Enter the source manager block.
  if (SLocEntryCursor.EnterSubBlock(pch::SOURCE_MANAGER_BLOCK_ID)) {
    Error("malformed source manager block record in PCH file");
    return Failure;
  }

  RecordData Record;
  while (true) {
    unsigned Code = SLocEntryCursor.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (SLocEntryCursor.ReadBlockEnd()) {
        Error("error at end of Source Manager block in PCH file");
        return Failure;
      }
      return Success;
    }
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      SLocEntryCursor.ReadSubBlockID();
      if (SLocEntryCursor.SkipBlock()) {
        Error("malformed block record in PCH file");
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

    case pch::SM_LINE_TABLE:
      if (ParseLineTable(Record))
        return Failure;
      break;

    case pch::SM_HEADER_FILE_INFO: {
      HeaderFileInfo HFI;
      HFI.isImport = Record[0];
      HFI.DirInfo = Record[1];
      HFI.NumIncludes = Record[2];
      HFI.ControllingMacroID = Record[3];
      if (Listener)
        Listener->ReadHeaderFileInfo(HFI);
      break;
    }

    case pch::SM_SLOC_FILE_ENTRY:
    case pch::SM_SLOC_BUFFER_ENTRY:
    case pch::SM_SLOC_INSTANTIATION_ENTRY:
      // Once we hit one of the source location entries, we're done.
      return Success;
    }
  }
}

/// \brief Read in the source location entry with the given ID.
PCHReader::PCHReadResult PCHReader::ReadSLocEntryRecord(unsigned ID) {
  if (ID == 0)
    return Success;

  if (ID > TotalNumSLocEntries) {
    Error("source location entry ID out-of-range for PCH file");
    return Failure;
  }

  ++NumSLocEntriesRead;
  SLocEntryCursor.JumpToBit(SLocOffsets[ID - 1]);
  unsigned Code = SLocEntryCursor.ReadCode();
  if (Code == llvm::bitc::END_BLOCK ||
      Code == llvm::bitc::ENTER_SUBBLOCK ||
      Code == llvm::bitc::DEFINE_ABBREV) {
    Error("incorrectly-formatted source location entry in PCH file");
    return Failure;
  }

  RecordData Record;
  const char *BlobStart;
  unsigned BlobLen;
  switch (SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen)) {
  default:
    Error("incorrectly-formatted source location entry in PCH file");
    return Failure;

  case pch::SM_SLOC_FILE_ENTRY: {
    std::string Filename(BlobStart, BlobStart + BlobLen);
    MaybeAddSystemRootToFilename(Filename);
    const FileEntry *File = FileMgr.getFile(Filename);
    if (File == 0) {
      std::string ErrorStr = "could not find file '";
      ErrorStr += Filename;
      ErrorStr += "' referenced by PCH file";
      Error(ErrorStr.c_str());
      return Failure;
    }
    
    FileID FID = SourceMgr.createFileID(File,
                                SourceLocation::getFromRawEncoding(Record[1]),
                                       (SrcMgr::CharacteristicKind)Record[2],
                                        ID, Record[0]);
    if (Record[3])
      const_cast<SrcMgr::FileInfo&>(SourceMgr.getSLocEntry(FID).getFile())
        .setHasLineDirectives();

    break;
  }

  case pch::SM_SLOC_BUFFER_ENTRY: {
    const char *Name = BlobStart;
    unsigned Offset = Record[0];
    unsigned Code = SLocEntryCursor.ReadCode();
    Record.clear();
    unsigned RecCode 
      = SLocEntryCursor.ReadRecord(Code, Record, &BlobStart, &BlobLen);
    assert(RecCode == pch::SM_SLOC_BUFFER_BLOB && "Ill-formed PCH file");
    (void)RecCode;
    llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBuffer(BlobStart, 
                                         BlobStart + BlobLen - 1,
                                         Name);
    FileID BufferID = SourceMgr.createFileIDForMemBuffer(Buffer, ID, Offset);
      
    if (strcmp(Name, "<built-in>") == 0) {
      PCHPredefinesBufferID = BufferID;
      PCHPredefines = BlobStart;
      PCHPredefinesLen = BlobLen - 1;
    }

    break;
  }

  case pch::SM_SLOC_INSTANTIATION_ENTRY: {
    SourceLocation SpellingLoc 
      = SourceLocation::getFromRawEncoding(Record[1]);
    SourceMgr.createInstantiationLoc(SpellingLoc,
                              SourceLocation::getFromRawEncoding(Record[2]),
                              SourceLocation::getFromRawEncoding(Record[3]),
                                     Record[4],
                                     ID,
                                     Record[0]);
    break;
  }  
  }

  return Success;
}

/// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
/// specified cursor.  Read the abbreviations that are at the top of the block
/// and then leave the cursor pointing into the block.
bool PCHReader::ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor,
                                 unsigned BlockID) {
  if (Cursor.EnterSubBlock(BlockID)) {
    Error("malformed block record in PCH file");
    return Failure;
  }
  
  while (true) {
    unsigned Code = Cursor.ReadCode();
    
    // We expect all abbrevs to be at the start of the block.
    if (Code != llvm::bitc::DEFINE_ABBREV)
      return false;
    Cursor.ReadAbbrevRecord();
  }
}

void PCHReader::ReadMacroRecord(uint64_t Offset) {
  assert(PP && "Forgot to set Preprocessor ?");
  
  // Keep track of where we are in the stream, then jump back there
  // after reading this macro.
  SavedStreamPosition SavedPosition(Stream);

  Stream.JumpToBit(Offset);
  RecordData Record;
  llvm::SmallVector<IdentifierInfo*, 16> MacroArgs;
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
        Error("malformed block record in PCH file");
        return;
      }
      continue;
    
    case llvm::bitc::DEFINE_ABBREV:
      Stream.ReadAbbrevRecord();
      continue;
    default: break;
    }

    // Read a record.
    Record.clear();
    pch::PreprocessorRecordTypes RecType =
      (pch::PreprocessorRecordTypes)Stream.ReadRecord(Code, Record);
    switch (RecType) {
    case pch::PP_MACRO_OBJECT_LIKE:
    case pch::PP_MACRO_FUNCTION_LIKE: {
      // If we already have a macro, that means that we've hit the end
      // of the definition of the macro we were looking for. We're
      // done.
      if (Macro)
        return;

      IdentifierInfo *II = DecodeIdentifierInfo(Record[0]);
      if (II == 0) {
        Error("macro must have a name in PCH file");
        return;
      }
      SourceLocation Loc = SourceLocation::getFromRawEncoding(Record[1]);
      bool isUsed = Record[2];
      
      MacroInfo *MI = PP->AllocateMacroInfo(Loc);
      MI->setIsUsed(isUsed);
      
      if (RecType == pch::PP_MACRO_FUNCTION_LIKE) {
        // Decode function-like macro info.
        bool isC99VarArgs = Record[3];
        bool isGNUVarArgs = Record[4];
        MacroArgs.clear();
        unsigned NumArgs = Record[5];
        for (unsigned i = 0; i != NumArgs; ++i)
          MacroArgs.push_back(DecodeIdentifierInfo(Record[6+i]));

        // Install function-like macro info.
        MI->setIsFunctionLike();
        if (isC99VarArgs) MI->setIsC99Varargs();
        if (isGNUVarArgs) MI->setIsGNUVarargs();
        MI->setArgumentList(MacroArgs.data(), MacroArgs.size(),
                            PP->getPreprocessorAllocator());
      }

      // Finally, install the macro.
      PP->setMacroInfo(II, MI);

      // Remember that we saw this macro last so that we add the tokens that
      // form its body to it.
      Macro = MI;
      ++NumMacrosRead;
      break;
    }
        
    case pch::PP_TOKEN: {
      // If we see a TOKEN before a PP_MACRO_*, then the file is
      // erroneous, just pretend we didn't see this.
      if (Macro == 0) break;
      
      Token Tok;
      Tok.startToken();
      Tok.setLocation(SourceLocation::getFromRawEncoding(Record[0]));
      Tok.setLength(Record[1]);
      if (IdentifierInfo *II = DecodeIdentifierInfo(Record[2]))
        Tok.setIdentifierInfo(II);
      Tok.setKind((tok::TokenKind)Record[3]);
      Tok.setFlag((Token::TokenFlags)Record[4]);
      Macro->AddTokenToBody(Tok);
      break;
    }
  }
  }
}

/// \brief If we are loading a relocatable PCH file, and the filename is
/// not an absolute path, add the system root to the beginning of the file
/// name.
void PCHReader::MaybeAddSystemRootToFilename(std::string &Filename) {
  // If this is not a relocatable PCH file, there's nothing to do.
  if (!RelocatablePCH)
    return;
  
  if (Filename.empty() || Filename[0] == '/' || Filename[0] == '<')
    return;

  std::string FIXME = Filename;
  
  if (isysroot == 0) {
    // If no system root was given, default to '/'
    Filename.insert(Filename.begin(), '/');
    return;
  }
  
  unsigned Length = strlen(isysroot);
  if (isysroot[Length - 1] != '/')
    Filename.insert(Filename.begin(), '/');
    
  Filename.insert(Filename.begin(), isysroot, isysroot + Length);
}

PCHReader::PCHReadResult 
PCHReader::ReadPCHBlock() {
  if (Stream.EnterSubBlock(pch::PCH_BLOCK_ID)) {
    Error("malformed block record in PCH file");
    return Failure;
  }

  // Read all of the records and blocks for the PCH file.
  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        Error("error at end of module block in PCH file");
        return Failure;
      }

      return Success;
    }

    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      case pch::TYPES_BLOCK_ID: // Skip types block (lazily loaded)
      default:  // Skip unknown content.
        if (Stream.SkipBlock()) {
          Error("malformed block record in PCH file");
          return Failure;
        }
        break;

      case pch::DECLS_BLOCK_ID:
        // We lazily load the decls block, but we want to set up the
        // DeclsCursor cursor to point into it.  Clone our current bitcode
        // cursor to it, enter the block and read the abbrevs in that block.
        // With the main cursor, we just skip over it.
        DeclsCursor = Stream;
        if (Stream.SkipBlock() ||  // Skip with the main cursor.
            // Read the abbrevs.
            ReadBlockAbbrevs(DeclsCursor, pch::DECLS_BLOCK_ID)) {
          Error("malformed block record in PCH file");
          return Failure;
        }
        break;
          
      case pch::PREPROCESSOR_BLOCK_ID:
        if (Stream.SkipBlock()) {
          Error("malformed block record in PCH file");
          return Failure;
        }
        break;

      case pch::SOURCE_MANAGER_BLOCK_ID:
        switch (ReadSourceManagerBlock()) {
        case Success:
          break;

        case Failure:
          Error("malformed source manager block in PCH file");
          return Failure;

        case IgnorePCH:
          return IgnorePCH;
        }
        break;
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
    switch ((pch::PCHRecordTypes)Stream.ReadRecord(Code, Record, 
                                                   &BlobStart, &BlobLen)) {
    default:  // Default behavior: ignore.
      break;

    case pch::TYPE_OFFSET:
      if (!TypesLoaded.empty()) {
        Error("duplicate TYPE_OFFSET record in PCH file");
        return Failure;
      }
      TypeOffsets = (const uint32_t *)BlobStart;
      TypesLoaded.resize(Record[0]);
      break;

    case pch::DECL_OFFSET:
      if (!DeclsLoaded.empty()) {
        Error("duplicate DECL_OFFSET record in PCH file");
        return Failure;
      }
      DeclOffsets = (const uint32_t *)BlobStart;
      DeclsLoaded.resize(Record[0]);
      break;

    case pch::LANGUAGE_OPTIONS:
      if (ParseLanguageOptions(Record))
        return IgnorePCH;
      break;

    case pch::METADATA: {
      if (Record[0] != pch::VERSION_MAJOR) {
        Diag(Record[0] < pch::VERSION_MAJOR? diag::warn_pch_version_too_old
                                           : diag::warn_pch_version_too_new);
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

    case pch::IDENTIFIER_TABLE:
      IdentifierTableData = BlobStart;
      if (Record[0]) {
        IdentifierLookupTable 
          = PCHIdentifierLookupTable::Create(
                        (const unsigned char *)IdentifierTableData + Record[0],
                        (const unsigned char *)IdentifierTableData, 
                        PCHIdentifierLookupTrait(*this));
        if (PP)
          PP->getIdentifierTable().setExternalIdentifierLookup(this);
      }
      break;

    case pch::IDENTIFIER_OFFSET:
      if (!IdentifiersLoaded.empty()) {
        Error("duplicate IDENTIFIER_OFFSET record in PCH file");
        return Failure;
      }
      IdentifierOffsets = (const uint32_t *)BlobStart;
      IdentifiersLoaded.resize(Record[0]);
      if (PP)
        PP->getHeaderSearchInfo().SetExternalLookup(this);
      break;

    case pch::EXTERNAL_DEFINITIONS:
      if (!ExternalDefinitions.empty()) {
        Error("duplicate EXTERNAL_DEFINITIONS record in PCH file");
        return Failure;
      }
      ExternalDefinitions.swap(Record);
      break;

    case pch::SPECIAL_TYPES:
      SpecialTypes.swap(Record);
      break;

    case pch::STATISTICS:
      TotalNumStatements = Record[0];
      TotalNumMacros = Record[1];
      TotalLexicalDeclContexts = Record[2];
      TotalVisibleDeclContexts = Record[3];
      break;

    case pch::TENTATIVE_DEFINITIONS:
      if (!TentativeDefinitions.empty()) {
        Error("duplicate TENTATIVE_DEFINITIONS record in PCH file");
        return Failure;
      }
      TentativeDefinitions.swap(Record);
      break;

    case pch::LOCALLY_SCOPED_EXTERNAL_DECLS:
      if (!LocallyScopedExternalDecls.empty()) {
        Error("duplicate LOCALLY_SCOPED_EXTERNAL_DECLS record in PCH file");
        return Failure;
      }
      LocallyScopedExternalDecls.swap(Record);
      break;

    case pch::SELECTOR_OFFSETS:
      SelectorOffsets = (const uint32_t *)BlobStart;
      TotalNumSelectors = Record[0];
      SelectorsLoaded.resize(TotalNumSelectors);
      break;

    case pch::METHOD_POOL:
      MethodPoolLookupTableData = (const unsigned char *)BlobStart;
      if (Record[0])
        MethodPoolLookupTable 
          = PCHMethodPoolLookupTable::Create(
                        MethodPoolLookupTableData + Record[0],
                        MethodPoolLookupTableData, 
                        PCHMethodPoolLookupTrait(*this));
      TotalSelectorsInMethodPool = Record[1];
      break;

    case pch::PP_COUNTER_VALUE:
      if (!Record.empty() && Listener)
        Listener->ReadCounter(Record[0]);
      break;

    case pch::SOURCE_LOCATION_OFFSETS:
      SLocOffsets = (const uint32_t *)BlobStart;
      TotalNumSLocEntries = Record[0];
      SourceMgr.PreallocateSLocEntries(this, 
                                                   TotalNumSLocEntries, 
                                                   Record[1]);
      break;

    case pch::SOURCE_LOCATION_PRELOADS:
      for (unsigned I = 0, N = Record.size(); I != N; ++I) {
        PCHReadResult Result = ReadSLocEntryRecord(Record[I]);
        if (Result != Success)
          return Result;
      }
      break;

    case pch::STAT_CACHE:
      FileMgr.setStatCache(
                  new PCHStatCache((const unsigned char *)BlobStart + Record[0],
                                   (const unsigned char *)BlobStart,
                                   NumStatHits, NumStatMisses));
      break;

    case pch::EXT_VECTOR_DECLS:
      if (!ExtVectorDecls.empty()) {
        Error("duplicate EXT_VECTOR_DECLS record in PCH file");
        return Failure;
      }
      ExtVectorDecls.swap(Record);
      break;

    case pch::ORIGINAL_FILE_NAME:
      OriginalFileName.assign(BlobStart, BlobLen);
      MaybeAddSystemRootToFilename(OriginalFileName);
      break;
        
    case pch::COMMENT_RANGES:
      Comments = (SourceRange *)BlobStart;
      NumComments = BlobLen / sizeof(SourceRange);
      break;
    }
  }
  Error("premature end of bitstream in PCH file");
  return Failure;
}

PCHReader::PCHReadResult PCHReader::ReadPCH(const std::string &FileName) {
  // Set the PCH file name.
  this->FileName = FileName;

  // Open the PCH file.
  std::string ErrStr;
  Buffer.reset(llvm::MemoryBuffer::getFile(FileName.c_str(), &ErrStr));
  if (!Buffer) {
    Error(ErrStr.c_str());
    return IgnorePCH;
  }

  // Initialize the stream
  StreamFile.init((const unsigned char *)Buffer->getBufferStart(), 
                  (const unsigned char *)Buffer->getBufferEnd());
  Stream.init(StreamFile);

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
      Error("invalid record at top-level of PCH file");
      return Failure;
    }

    unsigned BlockID = Stream.ReadSubBlockID();

    // We only know the PCH subblock ID.
    switch (BlockID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock()) {
        Error("malformed BlockInfoBlock in PCH file");
        return Failure;
      }
      break;
    case pch::PCH_BLOCK_ID:
      switch (ReadPCHBlock()) {
      case Success:
        break;

      case Failure:
        return Failure;

      case IgnorePCH:
        // FIXME: We could consider reading through to the end of this
        // PCH block, skipping subblocks, to see if there are other
        // PCH blocks elsewhere.

        // Clear out any preallocated source location entries, so that
        // the source manager does not try to resolve them later.
        SourceMgr.ClearPreallocatedSLocEntries();

        // Remove the stat cache.
        FileMgr.setStatCache(0);

        return IgnorePCH;
      }
      break;
    default:
      if (Stream.SkipBlock()) {
        Error("malformed block record in PCH file");
        return Failure;
      }
      break;
    }
  }  
  
  // Check the predefines buffer.
  if (CheckPredefinesBuffer(PCHPredefines, PCHPredefinesLen, 
                            PCHPredefinesBufferID))
    return IgnorePCH;
  
  if (PP) {
    // Initialization of keywords and pragmas occurs before the
    // PCH file is read, so there may be some identifiers that were
    // loaded into the IdentifierTable before we intercepted the
    // creation of identifiers. Iterate through the list of known
    // identifiers and determine whether we have to establish
    // preprocessor definitions or top-level identifier declaration
    // chains for those identifiers.
    //
    // We copy the IdentifierInfo pointers to a small vector first,
    // since de-serializing declarations or macro definitions can add
    // new entries into the identifier table, invalidating the
    // iterators.
    llvm::SmallVector<IdentifierInfo *, 128> Identifiers;
    for (IdentifierTable::iterator Id = PP->getIdentifierTable().begin(),
                                IdEnd = PP->getIdentifierTable().end();
         Id != IdEnd; ++Id)
      Identifiers.push_back(Id->second);
    PCHIdentifierLookupTable *IdTable 
      = (PCHIdentifierLookupTable *)IdentifierLookupTable;
    for (unsigned I = 0, N = Identifiers.size(); I != N; ++I) {
      IdentifierInfo *II = Identifiers[I];
      // Look in the on-disk hash table for an entry for
      PCHIdentifierLookupTrait Info(*this, II);
      std::pair<const char*, unsigned> Key(II->getName(), II->getLength());
      PCHIdentifierLookupTable::iterator Pos = IdTable->find(Key, &Info);
      if (Pos == IdTable->end())
        continue;
  
      // Dereferencing the iterator has the effect of populating the
      // IdentifierInfo node with the various declarations it needs.
      (void)*Pos;
    }
  }

  if (Context)
    InitializeContext(*Context);

  return Success;
}

void PCHReader::InitializeContext(ASTContext &Ctx) {
  Context = &Ctx;
  assert(Context && "Passed null context!");

  assert(PP && "Forgot to set Preprocessor ?");
  PP->getIdentifierTable().setExternalIdentifierLookup(this);
  PP->getHeaderSearchInfo().SetExternalLookup(this);
  
  // Load the translation unit declaration
  ReadDeclRecord(DeclOffsets[0], 0);

  // Load the special types.
  Context->setBuiltinVaListType(
    GetType(SpecialTypes[pch::SPECIAL_TYPE_BUILTIN_VA_LIST]));
  if (unsigned Id = SpecialTypes[pch::SPECIAL_TYPE_OBJC_ID])
    Context->setObjCIdType(GetType(Id));
  if (unsigned Sel = SpecialTypes[pch::SPECIAL_TYPE_OBJC_SELECTOR])
    Context->setObjCSelType(GetType(Sel));
  if (unsigned Proto = SpecialTypes[pch::SPECIAL_TYPE_OBJC_PROTOCOL])
    Context->setObjCProtoType(GetType(Proto));
  if (unsigned Class = SpecialTypes[pch::SPECIAL_TYPE_OBJC_CLASS])
    Context->setObjCClassType(GetType(Class));

  if (unsigned String = SpecialTypes[pch::SPECIAL_TYPE_CF_CONSTANT_STRING])
    Context->setCFConstantStringType(GetType(String));
  if (unsigned FastEnum 
        = SpecialTypes[pch::SPECIAL_TYPE_OBJC_FAST_ENUMERATION_STATE])
    Context->setObjCFastEnumerationStateType(GetType(FastEnum));
  if (unsigned File = SpecialTypes[pch::SPECIAL_TYPE_FILE]) {
    QualType FileType = GetType(File);
    assert(!FileType.isNull() && "FILE type is NULL");
    if (const TypedefType *Typedef = FileType->getAsTypedefType())
      Context->setFILEDecl(Typedef->getDecl());
    else {
      const TagType *Tag = FileType->getAs<TagType>();
      assert(Tag && "Invalid FILE type in PCH file");
      Context->setFILEDecl(Tag->getDecl());
    }
  }
  if (unsigned Jmp_buf = SpecialTypes[pch::SPECIAL_TYPE_jmp_buf]) {
    QualType Jmp_bufType = GetType(Jmp_buf);
    assert(!Jmp_bufType.isNull() && "jmp_bug type is NULL");
    if (const TypedefType *Typedef = Jmp_bufType->getAsTypedefType())
      Context->setjmp_bufDecl(Typedef->getDecl());
    else {
      const TagType *Tag = Jmp_bufType->getAs<TagType>();
      assert(Tag && "Invalid jmp_bug type in PCH file");
      Context->setjmp_bufDecl(Tag->getDecl());
    }
  }
  if (unsigned Sigjmp_buf = SpecialTypes[pch::SPECIAL_TYPE_sigjmp_buf]) {
    QualType Sigjmp_bufType = GetType(Sigjmp_buf);
    assert(!Sigjmp_bufType.isNull() && "sigjmp_buf type is NULL");
    if (const TypedefType *Typedef = Sigjmp_bufType->getAsTypedefType())
      Context->setsigjmp_bufDecl(Typedef->getDecl());
    else {
      const TagType *Tag = Sigjmp_bufType->getAs<TagType>();
      assert(Tag && "Invalid sigjmp_buf type in PCH file");
      Context->setsigjmp_bufDecl(Tag->getDecl());
    }
  }
  if (unsigned ObjCIdRedef 
        = SpecialTypes[pch::SPECIAL_TYPE_OBJC_ID_REDEFINITION])
    Context->ObjCIdRedefinitionType = GetType(ObjCIdRedef);
  if (unsigned ObjCClassRedef 
      = SpecialTypes[pch::SPECIAL_TYPE_OBJC_CLASS_REDEFINITION])
    Context->ObjCClassRedefinitionType = GetType(ObjCClassRedef);
}

/// \brief Retrieve the name of the original source file name
/// directly from the PCH file, without actually loading the PCH
/// file.
std::string PCHReader::getOriginalSourceFile(const std::string &PCHFileName) {
  // Open the PCH file.
  std::string ErrStr;
  llvm::OwningPtr<llvm::MemoryBuffer> Buffer;
  Buffer.reset(llvm::MemoryBuffer::getFile(PCHFileName.c_str(), &ErrStr));
  if (!Buffer) {
    fprintf(stderr, "error: %s\n", ErrStr.c_str());
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
    fprintf(stderr, 
            "error: '%s' does not appear to be a precompiled header file\n",
            PCHFileName.c_str());
    return std::string();
  }

  RecordData Record;
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    
    if (Code == llvm::bitc::ENTER_SUBBLOCK) {
      unsigned BlockID = Stream.ReadSubBlockID();
      
      // We only know the PCH subblock ID.
      switch (BlockID) {
      case pch::PCH_BLOCK_ID:
        if (Stream.EnterSubBlock(pch::PCH_BLOCK_ID)) {
          fprintf(stderr, "error: malformed block record in PCH file\n");
          return std::string();
        }
        break;
        
      default:
        if (Stream.SkipBlock()) {
          fprintf(stderr, "error: malformed block record in PCH file\n");
          return std::string();
        }
        break;
      }
      continue;
    }

    if (Code == llvm::bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd()) {
        fprintf(stderr, "error: error at end of module block in PCH file\n");
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
          == pch::ORIGINAL_FILE_NAME)
      return std::string(BlobStart, BlobLen);
  }  

  return std::string();
}

/// \brief Parse the record that corresponds to a LangOptions data
/// structure.
///
/// This routine compares the language options used to generate the
/// PCH file against the language options set for the current
/// compilation. For each option, we classify differences between the
/// two compiler states as either "benign" or "important". Benign
/// differences don't matter, and we accept them without complaint
/// (and without modifying the language options). Differences between
/// the states for important options cause the PCH file to be
/// unusable, so we emit a warning and return true to indicate that
/// there was an error.
///
/// \returns true if the PCH file is unacceptable, false otherwise.
bool PCHReader::ParseLanguageOptions(
                             const llvm::SmallVectorImpl<uint64_t> &Record) {
  if (Listener) {
    LangOptions LangOpts;
    
  #define PARSE_LANGOPT(Option)                  \
      LangOpts.Option = Record[Idx];             \
      ++Idx
    
    unsigned Idx = 0;
    PARSE_LANGOPT(Trigraphs);
    PARSE_LANGOPT(BCPLComment);
    PARSE_LANGOPT(DollarIdents);
    PARSE_LANGOPT(AsmPreprocessor);
    PARSE_LANGOPT(GNUMode);
    PARSE_LANGOPT(ImplicitInt);
    PARSE_LANGOPT(Digraphs);
    PARSE_LANGOPT(HexFloats);
    PARSE_LANGOPT(C99);
    PARSE_LANGOPT(Microsoft);
    PARSE_LANGOPT(CPlusPlus);
    PARSE_LANGOPT(CPlusPlus0x);
    PARSE_LANGOPT(CXXOperatorNames);
    PARSE_LANGOPT(ObjC1);
    PARSE_LANGOPT(ObjC2);
    PARSE_LANGOPT(ObjCNonFragileABI);
    PARSE_LANGOPT(PascalStrings);
    PARSE_LANGOPT(WritableStrings);
    PARSE_LANGOPT(LaxVectorConversions);
    PARSE_LANGOPT(AltiVec);
    PARSE_LANGOPT(Exceptions);
    PARSE_LANGOPT(NeXTRuntime);
    PARSE_LANGOPT(Freestanding);
    PARSE_LANGOPT(NoBuiltin);
    PARSE_LANGOPT(ThreadsafeStatics);
    PARSE_LANGOPT(Blocks);
    PARSE_LANGOPT(EmitAllDecls);
    PARSE_LANGOPT(MathErrno);
    PARSE_LANGOPT(OverflowChecking);
    PARSE_LANGOPT(HeinousExtensions);
    PARSE_LANGOPT(Optimize);
    PARSE_LANGOPT(OptimizeSize);
    PARSE_LANGOPT(Static);
    PARSE_LANGOPT(PICLevel);
    PARSE_LANGOPT(GNUInline);
    PARSE_LANGOPT(NoInline);
    PARSE_LANGOPT(AccessControl);
    PARSE_LANGOPT(CharIsSigned);
    LangOpts.setGCMode((LangOptions::GCMode)Record[Idx]);
    ++Idx;
    LangOpts.setVisibilityMode((LangOptions::VisibilityMode)Record[Idx]);
    ++Idx;
    PARSE_LANGOPT(InstantiationDepth);
    PARSE_LANGOPT(OpenCL);
  #undef PARSE_LANGOPT

    return Listener->ReadLanguageOptions(LangOpts);
  }

  return false;
}

void PCHReader::ReadComments(std::vector<SourceRange> &Comments) {
  Comments.resize(NumComments);
  std::copy(this->Comments, this->Comments + NumComments,
            Comments.begin());
}

/// \brief Read and return the type at the given offset.
///
/// This routine actually reads the record corresponding to the type
/// at the given offset in the bitstream. It is a helper routine for
/// GetType, which deals with reading type IDs.
QualType PCHReader::ReadTypeRecord(uint64_t Offset) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this type.
  SavedStreamPosition SavedPosition(Stream);

  // Note that we are loading a type record.
  LoadingTypeOrDecl Loading(*this);
  
  Stream.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = Stream.ReadCode();
  switch ((pch::TypeCode)Stream.ReadRecord(Code, Record)) {
  case pch::TYPE_EXT_QUAL: {
    assert(Record.size() == 3 && 
           "Incorrect encoding of extended qualifier type");
    QualType Base = GetType(Record[0]);
    QualType::GCAttrTypes GCAttr = (QualType::GCAttrTypes)Record[1];
    unsigned AddressSpace = Record[2];
    
    QualType T = Base;
    if (GCAttr != QualType::GCNone)
      T = Context->getObjCGCQualType(T, GCAttr);
    if (AddressSpace)
      T = Context->getAddrSpaceQualType(T, AddressSpace);
    return T;
  }

  case pch::TYPE_FIXED_WIDTH_INT: {
    assert(Record.size() == 2 && "Incorrect encoding of fixed-width int type");
    return Context->getFixedWidthIntType(Record[0], Record[1]);
  }

  case pch::TYPE_COMPLEX: {
    assert(Record.size() == 1 && "Incorrect encoding of complex type");
    QualType ElemType = GetType(Record[0]);
    return Context->getComplexType(ElemType);
  }

  case pch::TYPE_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context->getPointerType(PointeeType);
  }

  case pch::TYPE_BLOCK_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of block pointer type");
    QualType PointeeType = GetType(Record[0]);
    return Context->getBlockPointerType(PointeeType);
  }

  case pch::TYPE_LVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of lvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context->getLValueReferenceType(PointeeType);
  }

  case pch::TYPE_RVALUE_REFERENCE: {
    assert(Record.size() == 1 && "Incorrect encoding of rvalue reference type");
    QualType PointeeType = GetType(Record[0]);
    return Context->getRValueReferenceType(PointeeType);
  }

  case pch::TYPE_MEMBER_POINTER: {
    assert(Record.size() == 1 && "Incorrect encoding of member pointer type");
    QualType PointeeType = GetType(Record[0]);
    QualType ClassType = GetType(Record[1]);
    return Context->getMemberPointerType(PointeeType, ClassType.getTypePtr());
  }

  case pch::TYPE_CONSTANT_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    unsigned Idx = 3;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context->getConstantArrayType(ElementType, Size,
                                         ASM, IndexTypeQuals);
  }

  case pch::TYPE_CONSTANT_ARRAY_WITH_EXPR: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    SourceLocation LBLoc = SourceLocation::getFromRawEncoding(Record[3]);
    SourceLocation RBLoc = SourceLocation::getFromRawEncoding(Record[4]);
    unsigned Idx = 5;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context->getConstantArrayWithExprType(ElementType,
                                                 Size, ReadTypeExpr(),
                                                 ASM, IndexTypeQuals,
                                                 SourceRange(LBLoc, RBLoc));
  }

  case pch::TYPE_CONSTANT_ARRAY_WITHOUT_EXPR: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    unsigned Idx = 3;
    llvm::APInt Size = ReadAPInt(Record, Idx);
    return Context->getConstantArrayWithoutExprType(ElementType, Size,
                                                    ASM, IndexTypeQuals);
  }

  case pch::TYPE_INCOMPLETE_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    return Context->getIncompleteArrayType(ElementType, ASM, IndexTypeQuals);
  }

  case pch::TYPE_VARIABLE_ARRAY: {
    QualType ElementType = GetType(Record[0]);
    ArrayType::ArraySizeModifier ASM = (ArrayType::ArraySizeModifier)Record[1];
    unsigned IndexTypeQuals = Record[2];
    SourceLocation LBLoc = SourceLocation::getFromRawEncoding(Record[3]);
    SourceLocation RBLoc = SourceLocation::getFromRawEncoding(Record[4]);
    return Context->getVariableArrayType(ElementType, ReadTypeExpr(),
                                         ASM, IndexTypeQuals,
                                         SourceRange(LBLoc, RBLoc));
  }

  case pch::TYPE_VECTOR: {
    if (Record.size() != 2) {
      Error("incorrect encoding of vector type in PCH file");
      return QualType();
    }

    QualType ElementType = GetType(Record[0]);
    unsigned NumElements = Record[1];
    return Context->getVectorType(ElementType, NumElements);
  }

  case pch::TYPE_EXT_VECTOR: {
    if (Record.size() != 2) {
      Error("incorrect encoding of extended vector type in PCH file");
      return QualType();
    }

    QualType ElementType = GetType(Record[0]);
    unsigned NumElements = Record[1];
    return Context->getExtVectorType(ElementType, NumElements);
  }

  case pch::TYPE_FUNCTION_NO_PROTO: {
    if (Record.size() != 1) {
      Error("incorrect encoding of no-proto function type");
      return QualType();
    }
    QualType ResultType = GetType(Record[0]);
    return Context->getFunctionNoProtoType(ResultType);
  }

  case pch::TYPE_FUNCTION_PROTO: {
    QualType ResultType = GetType(Record[0]);
    unsigned Idx = 1;
    unsigned NumParams = Record[Idx++];
    llvm::SmallVector<QualType, 16> ParamTypes;
    for (unsigned I = 0; I != NumParams; ++I)
      ParamTypes.push_back(GetType(Record[Idx++]));
    bool isVariadic = Record[Idx++];
    unsigned Quals = Record[Idx++];
    bool hasExceptionSpec = Record[Idx++];
    bool hasAnyExceptionSpec = Record[Idx++];
    unsigned NumExceptions = Record[Idx++];
    llvm::SmallVector<QualType, 2> Exceptions;
    for (unsigned I = 0; I != NumExceptions; ++I)
      Exceptions.push_back(GetType(Record[Idx++]));
    return Context->getFunctionType(ResultType, ParamTypes.data(), NumParams,
                                    isVariadic, Quals, hasExceptionSpec,
                                    hasAnyExceptionSpec, NumExceptions,
                                    Exceptions.data());
  }

  case pch::TYPE_TYPEDEF:
    assert(Record.size() == 1 && "incorrect encoding of typedef type");
    return Context->getTypeDeclType(cast<TypedefDecl>(GetDecl(Record[0])));

  case pch::TYPE_TYPEOF_EXPR:
    return Context->getTypeOfExprType(ReadTypeExpr());

  case pch::TYPE_TYPEOF: {
    if (Record.size() != 1) {
      Error("incorrect encoding of typeof(type) in PCH file");
      return QualType();
    }
    QualType UnderlyingType = GetType(Record[0]);
    return Context->getTypeOfType(UnderlyingType);
  }
   
  case pch::TYPE_DECLTYPE:
    return Context->getDecltypeType(ReadTypeExpr());

  case pch::TYPE_RECORD:
    assert(Record.size() == 1 && "incorrect encoding of record type");
    return Context->getTypeDeclType(cast<RecordDecl>(GetDecl(Record[0])));

  case pch::TYPE_ENUM:
    assert(Record.size() == 1 && "incorrect encoding of enum type");
    return Context->getTypeDeclType(cast<EnumDecl>(GetDecl(Record[0])));

  case pch::TYPE_OBJC_INTERFACE: {
    unsigned Idx = 0;
    ObjCInterfaceDecl *ItfD = cast<ObjCInterfaceDecl>(GetDecl(Record[Idx++]));
    unsigned NumProtos = Record[Idx++];
    llvm::SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(cast<ObjCProtocolDecl>(GetDecl(Record[Idx++])));
    return Context->getObjCInterfaceType(ItfD, Protos.data(), NumProtos);
  }

  case pch::TYPE_OBJC_OBJECT_POINTER: {
    unsigned Idx = 0;
    QualType OIT = GetType(Record[Idx++]);
    unsigned NumProtos = Record[Idx++];
    llvm::SmallVector<ObjCProtocolDecl*, 4> Protos;
    for (unsigned I = 0; I != NumProtos; ++I)
      Protos.push_back(cast<ObjCProtocolDecl>(GetDecl(Record[Idx++])));
    return Context->getObjCObjectPointerType(OIT, Protos.data(), NumProtos);
  }
  }
  // Suppress a GCC warning
  return QualType();
}


QualType PCHReader::GetType(pch::TypeID ID) {
  unsigned Quals = ID & 0x07; 
  unsigned Index = ID >> 3;

  if (Index < pch::NUM_PREDEF_TYPE_IDS) {
    QualType T;
    switch ((pch::PredefinedTypeIDs)Index) {
    case pch::PREDEF_TYPE_NULL_ID: return QualType();
    case pch::PREDEF_TYPE_VOID_ID: T = Context->VoidTy; break;
    case pch::PREDEF_TYPE_BOOL_ID: T = Context->BoolTy; break;

    case pch::PREDEF_TYPE_CHAR_U_ID:
    case pch::PREDEF_TYPE_CHAR_S_ID:
      // FIXME: Check that the signedness of CharTy is correct!
      T = Context->CharTy;
      break;

    case pch::PREDEF_TYPE_UCHAR_ID:      T = Context->UnsignedCharTy;     break;
    case pch::PREDEF_TYPE_USHORT_ID:     T = Context->UnsignedShortTy;    break;
    case pch::PREDEF_TYPE_UINT_ID:       T = Context->UnsignedIntTy;      break;
    case pch::PREDEF_TYPE_ULONG_ID:      T = Context->UnsignedLongTy;     break;
    case pch::PREDEF_TYPE_ULONGLONG_ID:  T = Context->UnsignedLongLongTy; break;
    case pch::PREDEF_TYPE_UINT128_ID:    T = Context->UnsignedInt128Ty;   break;
    case pch::PREDEF_TYPE_SCHAR_ID:      T = Context->SignedCharTy;       break;
    case pch::PREDEF_TYPE_WCHAR_ID:      T = Context->WCharTy;            break;
    case pch::PREDEF_TYPE_SHORT_ID:      T = Context->ShortTy;            break;
    case pch::PREDEF_TYPE_INT_ID:        T = Context->IntTy;              break;
    case pch::PREDEF_TYPE_LONG_ID:       T = Context->LongTy;             break;
    case pch::PREDEF_TYPE_LONGLONG_ID:   T = Context->LongLongTy;         break;
    case pch::PREDEF_TYPE_INT128_ID:     T = Context->Int128Ty;           break;
    case pch::PREDEF_TYPE_FLOAT_ID:      T = Context->FloatTy;            break;
    case pch::PREDEF_TYPE_DOUBLE_ID:     T = Context->DoubleTy;           break;
    case pch::PREDEF_TYPE_LONGDOUBLE_ID: T = Context->LongDoubleTy;       break;
    case pch::PREDEF_TYPE_OVERLOAD_ID:   T = Context->OverloadTy;         break;
    case pch::PREDEF_TYPE_DEPENDENT_ID:  T = Context->DependentTy;        break;
    case pch::PREDEF_TYPE_NULLPTR_ID:    T = Context->NullPtrTy;          break;
    case pch::PREDEF_TYPE_CHAR16_ID:     T = Context->Char16Ty;           break;
    case pch::PREDEF_TYPE_CHAR32_ID:     T = Context->Char32Ty;           break;
    case pch::PREDEF_TYPE_OBJC_ID:       T = Context->ObjCBuiltinIdTy;    break;
    case pch::PREDEF_TYPE_OBJC_CLASS:    T = Context->ObjCBuiltinClassTy; break;
    }

    assert(!T.isNull() && "Unknown predefined type");
    return T.getQualifiedType(Quals);
  }

  Index -= pch::NUM_PREDEF_TYPE_IDS;
  //assert(Index < TypesLoaded.size() && "Type index out-of-range");
  if (!TypesLoaded[Index])
    TypesLoaded[Index] = ReadTypeRecord(TypeOffsets[Index]).getTypePtr();
    
  return QualType(TypesLoaded[Index], Quals);
}

Decl *PCHReader::GetDecl(pch::DeclID ID) {
  if (ID == 0)
    return 0;

  if (ID > DeclsLoaded.size()) {
    Error("declaration ID out-of-range for PCH file");
    return 0;
  }

  unsigned Index = ID - 1;
  if (!DeclsLoaded[Index])
    ReadDeclRecord(DeclOffsets[Index], Index);

  return DeclsLoaded[Index];
}

/// \brief Resolve the offset of a statement into a statement.
///
/// This operation will read a new statement from the external
/// source each time it is called, and is meant to be used via a
/// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
Stmt *PCHReader::GetDeclStmt(uint64_t Offset) {
  // Since we know tha this statement is part of a decl, make sure to use the
  // decl cursor to read it.
  DeclsCursor.JumpToBit(Offset);
  return ReadStmt(DeclsCursor);
}

bool PCHReader::ReadDeclsLexicallyInContext(DeclContext *DC,
                                  llvm::SmallVectorImpl<pch::DeclID> &Decls) {
  assert(DC->hasExternalLexicalStorage() && 
         "DeclContext has no lexical decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].first;
  assert(Offset && "DeclContext has no lexical decls in storage");

  // Keep track of where we are in the stream, then jump back there
  // after reading this context.
  SavedStreamPosition SavedPosition(DeclsCursor);

  // Load the record containing all of the declarations lexically in
  // this context.
  DeclsCursor.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned RecCode = DeclsCursor.ReadRecord(Code, Record);
  (void)RecCode;
  assert(RecCode == pch::DECL_CONTEXT_LEXICAL && "Expected lexical block");

  // Load all of the declaration IDs
  Decls.clear();
  Decls.insert(Decls.end(), Record.begin(), Record.end());
  ++NumLexicalDeclContextsRead;
  return false;
}

bool PCHReader::ReadDeclsVisibleInContext(DeclContext *DC,
                           llvm::SmallVectorImpl<VisibleDeclaration> &Decls) {
  assert(DC->hasExternalVisibleStorage() && 
         "DeclContext has no visible decls in storage");
  uint64_t Offset = DeclContextOffsets[DC].second;
  assert(Offset && "DeclContext has no visible decls in storage");

  // Keep track of where we are in the stream, then jump back there
  // after reading this context.
  SavedStreamPosition SavedPosition(DeclsCursor);

  // Load the record containing all of the declarations visible in
  // this context.
  DeclsCursor.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned RecCode = DeclsCursor.ReadRecord(Code, Record);
  (void)RecCode;
  assert(RecCode == pch::DECL_CONTEXT_VISIBLE && "Expected visible block");
  if (Record.size() == 0)
    return false;  

  Decls.clear();

  unsigned Idx = 0;
  while (Idx < Record.size()) {
    Decls.push_back(VisibleDeclaration());
    Decls.back().Name = ReadDeclarationName(Record, Idx);

    unsigned Size = Record[Idx++];
    llvm::SmallVector<unsigned, 4> &LoadedDecls = Decls.back().Declarations;
    LoadedDecls.reserve(Size);
    for (unsigned I = 0; I < Size; ++I)
      LoadedDecls.push_back(Record[Idx++]);
  }

  ++NumVisibleDeclContextsRead;
  return false;
}

void PCHReader::StartTranslationUnit(ASTConsumer *Consumer) {
  this->Consumer = Consumer;

  if (!Consumer)
    return;

  for (unsigned I = 0, N = ExternalDefinitions.size(); I != N; ++I) {
    Decl *D = GetDecl(ExternalDefinitions[I]);
    DeclGroupRef DG(D);
    Consumer->HandleTopLevelDecl(DG);
  }

  for (unsigned I = 0, N = InterestingDecls.size(); I != N; ++I) {
    DeclGroupRef DG(InterestingDecls[I]);
    Consumer->HandleTopLevelDecl(DG);
  }
}

void PCHReader::PrintStats() {
  std::fprintf(stderr, "*** PCH Statistics:\n");

  unsigned NumTypesLoaded 
    = TypesLoaded.size() - std::count(TypesLoaded.begin(), TypesLoaded.end(),
                                      (Type *)0);
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
  if (TotalNumSLocEntries)
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
  if (TotalNumSelectors)
    std::fprintf(stderr, "  %u/%u selectors read (%f%%)\n",
                 NumSelectorsLoaded, TotalNumSelectors,
                 ((float)NumSelectorsLoaded/TotalNumSelectors * 100));
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
  if (TotalSelectorsInMethodPool) {
    std::fprintf(stderr, "  %u/%u method pool entries read (%f%%)\n",
                 NumMethodPoolSelectorsRead, TotalSelectorsInMethodPool,
                 ((float)NumMethodPoolSelectorsRead/TotalSelectorsInMethodPool
                  * 100));
    std::fprintf(stderr, "  %u method pool misses\n", NumMethodPoolMisses);
  }
  std::fprintf(stderr, "\n");
}

void PCHReader::InitializeSema(Sema &S) {
  SemaObj = &S;
  S.ExternalSource = this;

  // Makes sure any declarations that were deserialized "too early"
  // still get added to the identifier's declaration chains.
  for (unsigned I = 0, N = PreloadedDecls.size(); I != N; ++I) {
    SemaObj->TUScope->AddDecl(Action::DeclPtrTy::make(PreloadedDecls[I]));
    SemaObj->IdResolver.AddDecl(PreloadedDecls[I]);
  }
  PreloadedDecls.clear();

  // If there were any tentative definitions, deserialize them and add
  // them to Sema's table of tentative definitions.
  for (unsigned I = 0, N = TentativeDefinitions.size(); I != N; ++I) {
    VarDecl *Var = cast<VarDecl>(GetDecl(TentativeDefinitions[I]));
    SemaObj->TentativeDefinitions[Var->getDeclName()] = Var;
    SemaObj->TentativeDefinitionList.push_back(Var->getDeclName());
  }

  // If there were any locally-scoped external declarations,
  // deserialize them and add them to Sema's table of locally-scoped
  // external declarations.
  for (unsigned I = 0, N = LocallyScopedExternalDecls.size(); I != N; ++I) {
    NamedDecl *D = cast<NamedDecl>(GetDecl(LocallyScopedExternalDecls[I]));
    SemaObj->LocallyScopedExternalDecls[D->getDeclName()] = D;
  }

  // If there were any ext_vector type declarations, deserialize them
  // and add them to Sema's vector of such declarations.
  for (unsigned I = 0, N = ExtVectorDecls.size(); I != N; ++I)
    SemaObj->ExtVectorDecls.push_back(
                               cast<TypedefDecl>(GetDecl(ExtVectorDecls[I])));
}

IdentifierInfo* PCHReader::get(const char *NameStart, const char *NameEnd) {
  // Try to find this name within our on-disk hash table
  PCHIdentifierLookupTable *IdTable 
    = (PCHIdentifierLookupTable *)IdentifierLookupTable;
  std::pair<const char*, unsigned> Key(NameStart, NameEnd - NameStart);
  PCHIdentifierLookupTable::iterator Pos = IdTable->find(Key);
  if (Pos == IdTable->end())
    return 0;

  // Dereferencing the iterator has the effect of building the
  // IdentifierInfo node and populating it with the various
  // declarations it needs.
  return *Pos;
}

std::pair<ObjCMethodList, ObjCMethodList> 
PCHReader::ReadMethodPool(Selector Sel) {
  if (!MethodPoolLookupTable)
    return std::pair<ObjCMethodList, ObjCMethodList>();

  // Try to find this selector within our on-disk hash table.
  PCHMethodPoolLookupTable *PoolTable
    = (PCHMethodPoolLookupTable*)MethodPoolLookupTable;
  PCHMethodPoolLookupTable::iterator Pos = PoolTable->find(Sel);
  if (Pos == PoolTable->end()) {
    ++NumMethodPoolMisses;
    return std::pair<ObjCMethodList, ObjCMethodList>();;
  }

  ++NumMethodPoolSelectorsRead;
  return *Pos;
}

void PCHReader::SetIdentifierInfo(unsigned ID, IdentifierInfo *II) {
  assert(ID && "Non-zero identifier ID required");
  assert(ID <= IdentifiersLoaded.size() && "identifier ID out of range");
  IdentifiersLoaded[ID - 1] = II;
}

/// \brief Set the globally-visible declarations associated with the given
/// identifier.
///
/// If the PCH reader is currently in a state where the given declaration IDs
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
PCHReader::SetGloballyVisibleDecls(IdentifierInfo *II, 
                              const llvm::SmallVectorImpl<uint32_t> &DeclIDs,
                                   bool Nonrecursive) {
  if (CurrentlyLoadingTypeOrDecl && !Nonrecursive) {
    PendingIdentifierInfos.push_back(PendingIdentifierInfo());
    PendingIdentifierInfo &PII = PendingIdentifierInfos.back();
    PII.II = II;
    for (unsigned I = 0, N = DeclIDs.size(); I != N; ++I)
      PII.DeclIDs.push_back(DeclIDs[I]);
    return;
  }
      
  for (unsigned I = 0, N = DeclIDs.size(); I != N; ++I) {
    NamedDecl *D = cast<NamedDecl>(GetDecl(DeclIDs[I]));
    if (SemaObj) {
      // Introduce this declaration into the translation-unit scope
      // and add it to the declaration chain for this identifier, so
      // that (unqualified) name lookup will find it.
      SemaObj->TUScope->AddDecl(Action::DeclPtrTy::make(D));
      SemaObj->IdResolver.AddDeclToIdentifierChain(II, D);
    } else {
      // Queue this declaration so that it will be added to the
      // translation unit scope and identifier's declaration chain
      // once a Sema object is known.
      PreloadedDecls.push_back(D);
    }
  }
}

IdentifierInfo *PCHReader::DecodeIdentifierInfo(unsigned ID) {
  if (ID == 0)
    return 0;
  
  if (!IdentifierTableData || IdentifiersLoaded.empty()) {
    Error("no identifier table in PCH file");
    return 0;
  }
  
  assert(PP && "Forgot to set Preprocessor ?");
  if (!IdentifiersLoaded[ID - 1]) {
    uint32_t Offset = IdentifierOffsets[ID - 1];
    const char *Str = IdentifierTableData + Offset;

    // All of the strings in the PCH file are preceded by a 16-bit
    // length. Extract that 16-bit length to avoid having to execute
    // strlen().
    const char *StrLenPtr = Str - 2;
    unsigned StrLen = (((unsigned) StrLenPtr[0])
                       | (((unsigned) StrLenPtr[1]) << 8)) - 1;
    IdentifiersLoaded[ID - 1] 
      = &PP->getIdentifierTable().get(Str, Str + StrLen);
  }
  
  return IdentifiersLoaded[ID - 1];
}

void PCHReader::ReadSLocEntry(unsigned ID) {
  ReadSLocEntryRecord(ID);
}

Selector PCHReader::DecodeSelector(unsigned ID) {
  if (ID == 0)
    return Selector();
  
  if (!MethodPoolLookupTableData)
    return Selector();

  if (ID > TotalNumSelectors) {
    Error("selector ID out of range in PCH file");
    return Selector();
  }

  unsigned Index = ID - 1;
  if (SelectorsLoaded[Index].getAsOpaquePtr() == 0) {
    // Load this selector from the selector table.
    // FIXME: endianness portability issues with SelectorOffsets table
    PCHMethodPoolLookupTrait Trait(*this);
    SelectorsLoaded[Index] 
      = Trait.ReadKey(MethodPoolLookupTableData + SelectorOffsets[Index], 0);
  }

  return SelectorsLoaded[Index];
}

DeclarationName 
PCHReader::ReadDeclarationName(const RecordData &Record, unsigned &Idx) {
  DeclarationName::NameKind Kind = (DeclarationName::NameKind)Record[Idx++];
  switch (Kind) {
  case DeclarationName::Identifier:
    return DeclarationName(GetIdentifierInfo(Record, Idx));

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return DeclarationName(GetSelector(Record, Idx));

  case DeclarationName::CXXConstructorName:
    return Context->DeclarationNames.getCXXConstructorName(
                          Context->getCanonicalType(GetType(Record[Idx++])));

  case DeclarationName::CXXDestructorName:
    return Context->DeclarationNames.getCXXDestructorName(
                          Context->getCanonicalType(GetType(Record[Idx++])));

  case DeclarationName::CXXConversionFunctionName:
    return Context->DeclarationNames.getCXXConversionFunctionName(
                          Context->getCanonicalType(GetType(Record[Idx++])));

  case DeclarationName::CXXOperatorName:
    return Context->DeclarationNames.getCXXOperatorName(
                                       (OverloadedOperatorKind)Record[Idx++]);

  case DeclarationName::CXXUsingDirective:
    return DeclarationName::getUsingDirectiveName();
  }

  // Required to silence GCC warning
  return DeclarationName();
}

/// \brief Read an integral value
llvm::APInt PCHReader::ReadAPInt(const RecordData &Record, unsigned &Idx) {
  unsigned BitWidth = Record[Idx++];
  unsigned NumWords = llvm::APInt::getNumWords(BitWidth);
  llvm::APInt Result(BitWidth, NumWords, &Record[Idx]);
  Idx += NumWords;
  return Result;
}

/// \brief Read a signed integral value
llvm::APSInt PCHReader::ReadAPSInt(const RecordData &Record, unsigned &Idx) {
  bool isUnsigned = Record[Idx++];
  return llvm::APSInt(ReadAPInt(Record, Idx), isUnsigned);
}

/// \brief Read a floating-point value
llvm::APFloat PCHReader::ReadAPFloat(const RecordData &Record, unsigned &Idx) {
  return llvm::APFloat(ReadAPInt(Record, Idx));
}

// \brief Read a string
std::string PCHReader::ReadString(const RecordData &Record, unsigned &Idx) {
  unsigned Len = Record[Idx++];
  std::string Result(Record.data() + Idx, Record.data() + Idx + Len);
  Idx += Len;
  return Result;
}

DiagnosticBuilder PCHReader::Diag(unsigned DiagID) {
  return Diag(SourceLocation(), DiagID);
}

DiagnosticBuilder PCHReader::Diag(SourceLocation Loc, unsigned DiagID) {
  return Diags.Report(FullSourceLoc(Loc, SourceMgr), DiagID);
}

/// \brief Retrieve the identifier table associated with the
/// preprocessor.
IdentifierTable &PCHReader::getIdentifierTable() {
  assert(PP && "Forgot to set Preprocessor ?");
  return PP->getIdentifierTable();
}

/// \brief Record that the given ID maps to the given switch-case
/// statement.
void PCHReader::RecordSwitchCaseID(SwitchCase *SC, unsigned ID) {
  assert(SwitchCaseStmts[ID] == 0 && "Already have a SwitchCase with this ID");
  SwitchCaseStmts[ID] = SC;
}

/// \brief Retrieve the switch-case statement with the given ID.
SwitchCase *PCHReader::getSwitchCaseWithID(unsigned ID) {
  assert(SwitchCaseStmts[ID] != 0 && "No SwitchCase with this ID");
  return SwitchCaseStmts[ID];
}

/// \brief Record that the given label statement has been
/// deserialized and has the given ID.
void PCHReader::RecordLabelStmt(LabelStmt *S, unsigned ID) {
  assert(LabelStmts.find(ID) == LabelStmts.end() && 
         "Deserialized label twice");
  LabelStmts[ID] = S;

  // If we've already seen any goto statements that point to this
  // label, resolve them now.
  typedef std::multimap<unsigned, GotoStmt *>::iterator GotoIter;
  std::pair<GotoIter, GotoIter> Gotos = UnresolvedGotoStmts.equal_range(ID);
  for (GotoIter Goto = Gotos.first; Goto != Gotos.second; ++Goto)
    Goto->second->setLabel(S);
  UnresolvedGotoStmts.erase(Gotos.first, Gotos.second);

  // If we've already seen any address-label statements that point to
  // this label, resolve them now.
  typedef std::multimap<unsigned, AddrLabelExpr *>::iterator AddrLabelIter;
  std::pair<AddrLabelIter, AddrLabelIter> AddrLabels 
    = UnresolvedAddrLabelExprs.equal_range(ID);
  for (AddrLabelIter AddrLabel = AddrLabels.first; 
       AddrLabel != AddrLabels.second; ++AddrLabel)
    AddrLabel->second->setLabel(S);
  UnresolvedAddrLabelExprs.erase(AddrLabels.first, AddrLabels.second);
}

/// \brief Set the label of the given statement to the label
/// identified by ID.
///
/// Depending on the order in which the label and other statements
/// referencing that label occur, this operation may complete
/// immediately (updating the statement) or it may queue the
/// statement to be back-patched later.
void PCHReader::SetLabelOf(GotoStmt *S, unsigned ID) {
  std::map<unsigned, LabelStmt *>::iterator Label = LabelStmts.find(ID);
  if (Label != LabelStmts.end()) {
    // We've already seen this label, so set the label of the goto and
    // we're done.
    S->setLabel(Label->second);
  } else {
    // We haven't seen this label yet, so add this goto to the set of
    // unresolved goto statements.
    UnresolvedGotoStmts.insert(std::make_pair(ID, S));
  }
}

/// \brief Set the label of the given expression to the label
/// identified by ID.
///
/// Depending on the order in which the label and other statements
/// referencing that label occur, this operation may complete
/// immediately (updating the statement) or it may queue the
/// statement to be back-patched later.
void PCHReader::SetLabelOf(AddrLabelExpr *S, unsigned ID) {
  std::map<unsigned, LabelStmt *>::iterator Label = LabelStmts.find(ID);
  if (Label != LabelStmts.end()) {
    // We've already seen this label, so set the label of the
    // label-address expression and we're done.
    S->setLabel(Label->second);
  } else {
    // We haven't seen this label yet, so add this label-address
    // expression to the set of unresolved label-address expressions.
    UnresolvedAddrLabelExprs.insert(std::make_pair(ID, S));
  }
}


PCHReader::LoadingTypeOrDecl::LoadingTypeOrDecl(PCHReader &Reader) 
  : Reader(Reader), Parent(Reader.CurrentlyLoadingTypeOrDecl) {
  Reader.CurrentlyLoadingTypeOrDecl = this;
}

PCHReader::LoadingTypeOrDecl::~LoadingTypeOrDecl() {
  if (!Parent) {
    // If any identifiers with corresponding top-level declarations have
    // been loaded, load those declarations now.
    while (!Reader.PendingIdentifierInfos.empty()) {
      Reader.SetGloballyVisibleDecls(Reader.PendingIdentifierInfos.front().II,
                                 Reader.PendingIdentifierInfos.front().DeclIDs,
                                     true);
      Reader.PendingIdentifierInfos.pop_front();
    }
  }

  Reader.CurrentlyLoadingTypeOrDecl = Parent;  
}
