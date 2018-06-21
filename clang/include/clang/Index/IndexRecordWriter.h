//===--- IndexRecordWriter.h - Index record serialization -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXRECORDWRITER_H
#define LLVM_CLANG_INDEX_INDEXRECORDWRITER_H

#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/SmallString.h"

namespace clang {
namespace index {

namespace writer {
/// An opaque pointer to a declaration or other symbol used by the
/// IndexRecordWriter to identify when two occurrences refer to the same symbol,
/// and as a token for getting information about a symbol from the caller.
typedef const void *OpaqueDecl;

/// An indexer symbol suitable for serialization.
///
/// This includes all the information about the symbol that will be serialized
/// except for roles, which are synthesized by looking at all the occurrences.
///
/// \seealso IndexRecordDecl
/// \note this struct is generally accompanied by a buffer that owns the string
/// storage.  It should not be stored permanently.
struct Symbol {
  SymbolInfo SymInfo;
  StringRef Name;
  StringRef USR;
  StringRef CodeGenName;
};

/// An relation to an opaque symbol.
/// \seealso IndexRecordRelation
struct SymbolRelation {
  OpaqueDecl RelatedSymbol;
  SymbolRoleSet Roles;
};

typedef llvm::function_ref<Symbol(OpaqueDecl, SmallVectorImpl<char> &Scratch)>
    SymbolWriterCallback;
} // end namespace writer

/// A language-independent utility for serializing index record files.
///
/// Internally, this class is a small state machine.  Users should first call
/// beginRecord, and if the file does not already exist, then proceed to add
/// all symbol occurrences (addOccurrence) and finally finish with endRecord.
class IndexRecordWriter {
  SmallString<64> RecordsPath; ///< The records directory path.
  void *Record = nullptr;      ///< The state of the current record.
public:
  IndexRecordWriter(StringRef IndexPath);

  enum class Result {
    Success,
    Failure,
    AlreadyExists,
  };

  /// Begin writing a record for the file \p Filename with contents uniquely
  /// identified by \p RecordHash.
  ///
  /// \param Filename the name of the file this is a record for.
  /// \param RecordHash the unique hash of the record contents.
  /// \param Error on failure, set to the error message.
  /// \param RecordFile if non-null, this is set to the name of the record file.
  ///
  /// \returns Success if we should continue writing this record, AlreadyExists
  /// if the record file has already been written, or Failure if there was an
  /// error, in which case \p Error will be set.
  Result beginRecord(StringRef Filename, llvm::hash_code RecordHash,
                     std::string &Error, std::string *RecordFile = nullptr);

  /// Finish writing the record file.
  ///
  /// \param Error on failure, set to the error message.
  /// \param GetSymbolForDecl a callback mapping an writer::OpaqueDecl to its
  /// writer::Symbol. This is how the language-specific symbol information is
  /// provided to the IndexRecordWriter. The scratch parameter can be used for
  /// any necessary storage.
  ///
  /// \return Success, or Failure and sets \p Error.
  Result endRecord(std::string &Error,
                   writer::SymbolWriterCallback GetSymbolForDecl);

  /// Add an occurrence of the symbol \p D with the given \p Roles and location.
  void addOccurrence(writer::OpaqueDecl D, SymbolRoleSet Roles, unsigned Line,
                     unsigned Column, ArrayRef<writer::SymbolRelation> Related);
};

} // end namespace index
} // end namespace clang

#endif // LLVM_CLANG_INDEX_INDEXRECORDWRITER_H
