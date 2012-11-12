//===--- SerializedDiagnosticPrinter.h - Serializer for diagnostics -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_SERIALIZE_DIAGNOSTIC_PRINTER_H_
#define LLVM_CLANG_FRONTEND_SERIALIZE_DIAGNOSTIC_PRINTER_H_

#include "llvm/Bitcode/BitstreamWriter.h"

namespace llvm {
class raw_ostream;
}

namespace clang {
class DiagnosticConsumer;
class DiagnosticsEngine;
class DiagnosticOptions;

namespace serialized_diags {
  
enum BlockIDs {
  /// \brief A top-level block which represents any meta data associated
  /// with the diagostics, including versioning of the format.
  BLOCK_META = llvm::bitc::FIRST_APPLICATION_BLOCKID,

  /// \brief The this block acts as a container for all the information
  /// for a specific diagnostic.
  BLOCK_DIAG
};

enum RecordIDs {
  RECORD_VERSION = 1,
  RECORD_DIAG,
  RECORD_SOURCE_RANGE,
  RECORD_DIAG_FLAG,
  RECORD_CATEGORY,
  RECORD_FILENAME,
  RECORD_FIXIT,
  RECORD_FIRST = RECORD_VERSION,
  RECORD_LAST = RECORD_FIXIT
};

/// \brief Returns a DiagnosticConsumer that serializes diagnostics to
///  a bitcode file.
///
/// The created DiagnosticConsumer is designed for quick and lightweight
/// transfer of of diagnostics to the enclosing build system (e.g., an IDE).
/// This allows wrapper tools for Clang to get diagnostics from Clang
/// (via libclang) without needing to parse Clang's command line output.
///
DiagnosticConsumer *create(llvm::raw_ostream *OS,
                           DiagnosticOptions *diags);

} // end serialized_diags namespace
} // end clang namespace

#endif
