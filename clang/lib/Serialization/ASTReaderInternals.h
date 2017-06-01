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
#ifndef LLVM_CLANG_LIB_SERIALIZATION_ASTREADERINTERNALS_H
#define LLVM_CLANG_LIB_SERIALIZATION_ASTREADERINTERNALS_H

#include "MultiOnDiskHashTable.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/OnDiskHashTable.h"
#include <utility>

namespace clang {

class ASTReader;
class HeaderSearch;
struct HeaderFileInfo;
class FileEntry;
  
namespace serialization {

class ModuleFile;

namespace reader {

/// \brief Class that performs name lookup into a DeclContext stored
/// in an AST file.
class ASTDeclContextNameLookupTrait {
  ASTReader &Reader;
  ModuleFile &F;
  
public:
  // Maximum number of lookup tables we allow before condensing the tables.
  static const int MaxTables = 4;

  /// The lookup result is a list of global declaration IDs.
  typedef llvm::SmallVector<DeclID, 4> data_type;
  struct data_type_builder {
    data_type &Data;
    llvm::DenseSet<DeclID> Found;

    data_type_builder(data_type &D) : Data(D) {}
    void insert(DeclID ID) {
      // Just use a linear scan unless we have more than a few IDs.
      if (Found.empty() && !Data.empty()) {
        if (Data.size() <= 4) {
          for (auto I : Found)
            if (I == ID)
              return;
          Data.push_back(ID);
          return;
        }

        // Switch to tracking found IDs in the set.
        Found.insert(Data.begin(), Data.end());
      }

      if (Found.insert(ID).second)
        Data.push_back(ID);
    }
  };
  typedef unsigned hash_value_type;
  typedef unsigned offset_type;
  typedef ModuleFile *file_type;

  typedef DeclarationName external_key_type;
  typedef DeclarationNameKey internal_key_type;

  explicit ASTDeclContextNameLookupTrait(ASTReader &Reader, ModuleFile &F)
    : Reader(Reader), F(F) { }

  static bool EqualKey(const internal_key_type &a, const internal_key_type &b) {
    return a == b;
  }

  static hash_value_type ComputeHash(const internal_key_type &Key) {
    return Key.getHash();
  }
  static internal_key_type GetInternalKey(const external_key_type &Name) {
    return Name;
  }

  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char *&d);

  internal_key_type ReadKey(const unsigned char *d, unsigned);

  void ReadDataInto(internal_key_type, const unsigned char *d,
                    unsigned DataLen, data_type_builder &Val);

  static void MergeDataInto(const data_type &From, data_type_builder &To) {
    To.Data.reserve(To.Data.size() + From.size());
    for (DeclID ID : From)
      To.insert(ID);
  }

  file_type ReadFileRef(const unsigned char *&d);
};

struct DeclContextLookupTable {
  MultiOnDiskHashTable<ASTDeclContextNameLookupTrait> Table;
};

/// \brief Base class for the trait describing the on-disk hash table for the
/// identifiers in an AST file.
///
/// This class is not useful by itself; rather, it provides common
/// functionality for accessing the on-disk hash table of identifiers
/// in an AST file. Different subclasses customize that functionality
/// based on what information they are interested in. Those subclasses
/// must provide the \c data_type typedef and the ReadData operation,
/// only.
class ASTIdentifierLookupTraitBase {
public:
  typedef StringRef external_key_type;
  typedef StringRef internal_key_type;
  typedef unsigned hash_value_type;
  typedef unsigned offset_type;

  static bool EqualKey(const internal_key_type& a, const internal_key_type& b) {
    return a == b;
  }

  static hash_value_type ComputeHash(const internal_key_type& a);
 
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);

  // This hopefully will just get inlined and removed by the optimizer.
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  // This hopefully will just get inlined and removed by the optimizer.
  static const external_key_type&
  GetExternalKey(const internal_key_type& x) { return x; }

  static internal_key_type ReadKey(const unsigned char* d, unsigned n); 
};

/// \brief Class that performs lookup for an identifier stored in an AST file.
class ASTIdentifierLookupTrait : public ASTIdentifierLookupTraitBase {
  ASTReader &Reader;
  ModuleFile &F;
  
  // If we know the IdentifierInfo in advance, it is here and we will
  // not build a new one. Used when deserializing information about an
  // identifier that was constructed before the AST file was read.
  IdentifierInfo *KnownII;
  
public:
  typedef IdentifierInfo * data_type;

  ASTIdentifierLookupTrait(ASTReader &Reader, ModuleFile &F,
                           IdentifierInfo *II = nullptr)
    : Reader(Reader), F(F), KnownII(II) { }

  data_type ReadData(const internal_key_type& k,
                     const unsigned char* d,
                     unsigned DataLen);
  
  IdentID ReadIdentifierID(const unsigned char *d);

  ASTReader &getReader() const { return Reader; }
};
  
/// \brief The on-disk hash table used to contain information about
/// all of the identifiers in the program.
typedef llvm::OnDiskIterableChainedHashTable<ASTIdentifierLookupTrait>
  ASTIdentifierLookupTable;

/// \brief Class that performs lookup for a selector's entries in the global
/// method pool stored in an AST file.
class ASTSelectorLookupTrait {
  ASTReader &Reader;
  ModuleFile &F;
  
public:
  struct data_type {
    SelectorID ID;
    unsigned InstanceBits;
    unsigned FactoryBits;
    bool InstanceHasMoreThanOneDecl;
    bool FactoryHasMoreThanOneDecl;
    SmallVector<ObjCMethodDecl *, 2> Instance;
    SmallVector<ObjCMethodDecl *, 2> Factory;
  };
  
  typedef Selector external_key_type;
  typedef external_key_type internal_key_type;
  typedef unsigned hash_value_type;
  typedef unsigned offset_type;
  
  ASTSelectorLookupTrait(ASTReader &Reader, ModuleFile &F) 
    : Reader(Reader), F(F) { }
  
  static bool EqualKey(const internal_key_type& a,
                       const internal_key_type& b) {
    return a == b;
  }
  
  static hash_value_type ComputeHash(Selector Sel);
  
  static const internal_key_type&
  GetInternalKey(const external_key_type& x) { return x; }
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);
  
  internal_key_type ReadKey(const unsigned char* d, unsigned);
  data_type ReadData(Selector, const unsigned char* d, unsigned DataLen);
};
  
/// \brief The on-disk hash table used for the global method pool.
typedef llvm::OnDiskChainedHashTable<ASTSelectorLookupTrait>
  ASTSelectorLookupTable;
  
/// \brief Trait class used to search the on-disk hash table containing all of
/// the header search information.
///
/// The on-disk hash table contains a mapping from each header path to 
/// information about that header (how many times it has been included, its
/// controlling macro, etc.). Note that we actually hash based on the size
/// and mtime, and support "deep" comparisons of file names based on current
/// inode numbers, so that the search can cope with non-normalized path names
/// and symlinks.
class HeaderFileInfoTrait {
  ASTReader &Reader;
  ModuleFile &M;
  HeaderSearch *HS;
  const char *FrameworkStrings;

public:
  typedef const FileEntry *external_key_type;

  struct internal_key_type {
    off_t Size;
    time_t ModTime;
    StringRef Filename;
    bool Imported;
  };
  typedef const internal_key_type &internal_key_ref;
  
  typedef HeaderFileInfo data_type;
  typedef unsigned hash_value_type;
  typedef unsigned offset_type;
  
  HeaderFileInfoTrait(ASTReader &Reader, ModuleFile &M, HeaderSearch *HS,
                      const char *FrameworkStrings)
  : Reader(Reader), M(M), HS(HS), FrameworkStrings(FrameworkStrings) { }
  
  static hash_value_type ComputeHash(internal_key_ref ikey);
  internal_key_type GetInternalKey(const FileEntry *FE);
  bool EqualKey(internal_key_ref a, internal_key_ref b);
  
  static std::pair<unsigned, unsigned>
  ReadKeyDataLength(const unsigned char*& d);
  
  static internal_key_type ReadKey(const unsigned char *d, unsigned);
  
  data_type ReadData(internal_key_ref,const unsigned char *d, unsigned DataLen);
};

/// \brief The on-disk hash table used for known header files.
typedef llvm::OnDiskChainedHashTable<HeaderFileInfoTrait>
  HeaderFileInfoLookupTable;
  
} // end namespace clang::serialization::reader
} // end namespace clang::serialization
} // end namespace clang


#endif
