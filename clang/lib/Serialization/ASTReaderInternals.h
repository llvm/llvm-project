//===--- ASTReaderInternals.h - AST Reader Internals ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides internal definitions used in the AST reader.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_ASTREADER_INTERNALS_H
#define LLVM_CLANG_SERIALIZATION_ASTREADER_INTERNALS_H

#include "clang/Basic/OnDiskHashTable.h"
#include "clang/AST/DeclarationName.h"
#include "llvm/Support/Endian.h"
#include <utility>
#include <sys/stat.h>

namespace clang {

class ASTReader;
class HeaderSearch;
struct HeaderFileInfo;
  
namespace serialization {

class ModuleFile;

namespace reader {

/// \brief Class that performs name lookup into a DeclContext stored
/// in an AST file.
class ASTDeclContextNameLookupTrait {
  ASTReader &Reader;
  ModuleFile &F;
  
public:
  /// \brief Pair of begin/end iterators for DeclIDs.
  ///
  /// Note that these declaration IDs are local to the module that contains this
  /// particular lookup t
  typedef llvm::support::ulittle32_t LE32DeclID;
  typedef std::pair<LE32DeclID *, LE32DeclID *> data_type;

  /// \brief Special internal key for declaration names.
  /// The hash table creates keys for comparison; we do not create
  /// a DeclarationName for the internal key to avoid deserializing types.
  struct DeclNameKey {
    DeclarationName::NameKind Kind;
    uint64_t Data;
    DeclNameKey() : Kind((DeclarationName::NameKind)0), Data(0) { }
  };

  typedef DeclarationName external_key_type;
  typedef DeclNameKey internal_key_type;

  explicit ASTDeclContextNameLookupTrait(ASTReader &Reader, ModuleFile &F)
    : Reader(Reader), F(F) { }

  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a.Kind == b.Kind && a.Data == b.Data;
  }

  unsigned ComputeHash(const DeclNameKey &Key) const;
  internal_key_type GetInternalKey(const external_key_type& Name) const;

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);

  internal_key_type ReadKey(const unsigned char* d, unsigned);

  data_type ReadData(internal_key_type, const unsigned char* d,
                     unsigned DataLen);
};

/// \brief Class that performs lookup for an identifier stored in an AST file.
class ASTIdentifierLookupTrait {
  ASTReader &Reader;
  ModuleFile &F;
  
  // If we know the IdentifierInfo in advance, it is here and we will
  // not build a new one. Used when deserializing information about an
  // identifier that was constructed before the AST file was read.
  IdentifierInfo *KnownII;
  
public:
  typedef IdentifierInfo * data_type;
  
  typedef const std::pair<const char*, unsigned> external_key_type;
  
  typedef external_key_type internal_key_type;
  
  ASTIdentifierLookupTrait(ASTReader &Reader, ModuleFile &F,
                           IdentifierInfo *II = 0)
    : Reader(Reader), F(F), KnownII(II) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return (a.second == b.second) ? memcmp(a.first, b.first, a.second) == 0
    : false;
  }
  
  static unsigned ComputeHash(const internal_key_type& a);
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const external_key_type&
  GetExternalKey(const internal_key_type& x) { return x; }
 
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);

  static std::pair<const char*, unsigned>
  ReadKey(const unsigned char* d, unsigned n);
  
  IdentifierInfo *ReadData(const internal_key_type& k,
                           const unsigned char* d,
                           unsigned DataLen);
  
  ASTReader &getReader() const { return Reader; }
  
};
  
/// \brief The on-disk hash table used to contain information about
/// all of the identifiers in the program.
typedef OnDiskChainedHashTable<ASTIdentifierLookupTrait>
  ASTIdentifierLookupTable;

/// \brief Class that performs lookup for a selector's entries in the global
/// method pool stored in an AST file.
class ASTSelectorLookupTrait {
  ASTReader &Reader;
  ModuleFile &F;
  
public:
  struct data_type {
    SelectorID ID;
    llvm::SmallVector<ObjCMethodDecl *, 2> Instance;
    llvm::SmallVector<ObjCMethodDecl *, 2> Factory;
  };
  
  typedef Selector external_key_type;
  typedef external_key_type internal_key_type;
  
  ASTSelectorLookupTrait(ASTReader &Reader, ModuleFile &F) 
    : Reader(Reader), F(F) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a == b;
  }
  
  static unsigned ComputeHash(Selector Sel);
  
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);
  
  internal_key_type ReadKey(const unsigned char* d, unsigned);
  data_type ReadData(Selector, const unsigned char* d, unsigned DataLen);
};
  
/// \brief The on-disk hash table used for the global method pool.
typedef OnDiskChainedHashTable<ASTSelectorLookupTrait>
  ASTSelectorLookupTable;
  
/// \brief Trait class used to search the on-disk hash table containing all of
/// the header search information.
///
/// The on-disk hash table contains a mapping from each header path to 
/// information about that header (how many times it has been included, its
/// controlling macro, etc.). Note that we actually hash based on the 
/// filename, and support "deep" comparisons of file names based on current
/// inode numbers, so that the search can cope with non-normalized path names
/// and symlinks.
class HeaderFileInfoTrait {
  ASTReader &Reader;
  ModuleFile &M;
  HeaderSearch *HS;
  const char *FrameworkStrings;
  const char *SearchPath;
  struct stat SearchPathStatBuf;
  llvm::Optional<int> SearchPathStatResult;
  
  int StatSimpleCache(const char *Path, struct stat *StatBuf) {
    if (Path == SearchPath) {
      if (!SearchPathStatResult)
        SearchPathStatResult = stat(Path, &SearchPathStatBuf);
      
      *StatBuf = SearchPathStatBuf;
      return *SearchPathStatResult;
    }
    
    return stat(Path, StatBuf);
  }
  
public:
  typedef const char *external_key_type;
  typedef const char *internal_key_type;
  
  typedef HeaderFileInfo data_type;
  
  HeaderFileInfoTrait(ASTReader &Reader, ModuleFile &M, HeaderSearch *HS,
                      const char *FrameworkStrings,
                      const char *SearchPath = 0) 
  : Reader(Reader), M(M), HS(HS), FrameworkStrings(FrameworkStrings), 
    SearchPath(SearchPath) { }
  
  static unsigned ComputeHash(const char *path);
  static internal_key_type GetInternalKey(const char *path);
  bool EqualKey(internal_key_type a, internal_key_type b);
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);
  
  static internal_key_type ReadKey(const unsigned char *d, unsigned) {
    return (const char *)d;
  }
  
  data_type ReadData(const internal_key_type, const unsigned char *d,
                     unsigned DataLen);
};

/// \brief The on-disk hash table used for known header files.
typedef OnDiskChainedHashTable<HeaderFileInfoTrait>
  HeaderFileInfoLookupTable;
  
} // end namespace clang::serialization::reader
} // end namespace clang::serialization
} // end namespace clang


#endif
