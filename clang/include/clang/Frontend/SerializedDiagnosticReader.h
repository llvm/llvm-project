//===--- SerializedDiagnosticReader.h - Reads diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SERIALIZED_DIAGNOSTIC_READER_H_
#define LLVM_CLANG_FRONTEND_SERIALIZED_DIAGNOSTIC_READER_H_

#include "clang/Basic/LLVM.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/ErrorOr.h"

namespace clang {
namespace serialized_diags {

enum class SDError {
  CouldNotLoad = 1,
  InvalidSignature,
  InvalidDiagnostics,
  MalformedTopLevelBlock,
  MalformedSubBlock,
  MalformedBlockInfoBlock,
  MalformedMetadataBlock,
  MalformedDiagnosticBlock,
  MalformedDiagnosticRecord,
  MissingVersion,
  VersionMismatch,
  UnsupportedConstruct,
  /// A generic error for subclass handlers that don't want or need to define
  /// their own error_category.
  HandlerFailed
};

const std::error_category &SDErrorCategory();

inline std::error_code make_error_code(SDError E) {
  return std::error_code(static_cast<int>(E), SDErrorCategory());
}

/// \brief A location that is represented in the serialized diagnostics.
struct Location {
  unsigned FileID;
  unsigned Line;
  unsigned Col;
  unsigned Offset;
  Location(unsigned FileID, unsigned Line, unsigned Col, unsigned Offset)
      : FileID(FileID), Line(Line), Col(Col), Offset(Offset) {}
};

/// \brief A base class that handles reading serialized diagnostics from a file.
///
/// Subclasses should override the visit* methods with their logic for handling
/// the various constructs that are found in serialized diagnostics.
class SerializedDiagnosticReader {
public:
  SerializedDiagnosticReader() {}
  virtual ~SerializedDiagnosticReader() {}

  /// \brief Read the diagnostics in \c File
  std::error_code readDiagnostics(StringRef File);

private:
  enum class Cursor;

  /// \brief Read to the next record or block to process.
  llvm::ErrorOr<Cursor> skipUntilRecordOrBlock(llvm::BitstreamCursor &Stream,
                                               unsigned &BlockOrRecordId);

  /// \brief Read a metadata block from \c Stream.
  std::error_code readMetaBlock(llvm::BitstreamCursor &Stream);

  /// \brief Read a diagnostic block from \c Stream.
  std::error_code readDiagnosticBlock(llvm::BitstreamCursor &Stream);

protected:
  /// \brief Visit the start of a diagnostic block.
  virtual std::error_code visitStartOfDiagnostic() {
    return std::error_code();
  }
  /// \brief Visit the end of a diagnostic block.
  virtual std::error_code visitEndOfDiagnostic() { return std::error_code(); }
  /// \brief Visit a category. This associates the category \c ID to a \c Name.
  virtual std::error_code visitCategoryRecord(unsigned ID, StringRef Name) {
    return std::error_code();
  }
  /// \brief Visit a flag. This associates the flag's \c ID to a \c Name.
  virtual std::error_code visitDiagFlagRecord(unsigned ID, StringRef Name) {
    return std::error_code();
  }
  /// \brief Visit a diagnostic.
  virtual std::error_code
  visitDiagnosticRecord(unsigned Severity, const Location &Location,
                        unsigned Category, unsigned Flag, StringRef Message) {
    return std::error_code();
  }
  /// \brief Visit a filename. This associates the file's \c ID to a \c Name.
  virtual std::error_code visitFilenameRecord(unsigned ID, unsigned Size,
                                              unsigned Timestamp,
                                              StringRef Name) {
    return std::error_code();
  }
  /// \brief Visit a fixit hint.
  virtual std::error_code
  visitFixitRecord(const Location &Start, const Location &End, StringRef Text) {
    return std::error_code();
  }
  /// \brief Visit a source range.
  virtual std::error_code visitSourceRangeRecord(const Location &Start,
                                                 const Location &End) {
    return std::error_code();
  }
  /// \brief Visit the version of the set of diagnostics.
  virtual std::error_code visitVersionRecord(unsigned Version) {
    return std::error_code();
  }
};

} // end serialized_diags namespace
} // end clang namespace

namespace std {
template <>
struct is_error_code_enum<clang::serialized_diags::SDError> : std::true_type {};
}

#endif
