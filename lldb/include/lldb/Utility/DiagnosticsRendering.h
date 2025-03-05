//===-- DiagnosticsRendering.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_DIAGNOSTICSRENDERING_H
#define LLDB_UTILITY_DIAGNOSTICSRENDERING_H

#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "llvm/Support/WithColor.h"

namespace lldb_private {

/// A compiler-independent representation of an \c
/// lldb_private::Diagnostic. Expression evaluation failures often
/// have more than one diagnostic that a UI layer might want to render
/// differently, for example to colorize it.
///
/// Running example:
///   (lldb) expr 1 + foo
///   error: <user expression 0>:1:3: use of undeclared identifier 'foo'
///   1 + foo
///       ^~~
struct DiagnosticDetail {
  /// A source location consisting of a file name and position.
  struct SourceLocation {
    /// \c "<user expression 0>" in the example above.
    FileSpec file;
    /// \c 1 in the example above.
    unsigned line = 0;
    /// \c 5 in the example above.
    uint16_t column = 0;
    /// \c 3 in the example above.
    uint16_t length = 0;
    /// Whether this source location should be surfaced to the
    /// user. For example, syntax errors diagnosed in LLDB's
    /// expression wrapper code have this set to true.
    bool hidden = false;
    /// Whether this source location refers to something the user
    /// typed as part of the command, i.e., if this qualifies for
    /// inline display, or if the source line would need to be echoed
    /// again for the message to make sense.
    bool in_user_input = false;
  };
  /// Contains this diagnostic's source location, if applicable.
  std::optional<SourceLocation> source_location;
  /// Contains \c eSeverityError in the example above.
  lldb::Severity severity = lldb::eSeverityInfo;
  /// Contains "use of undeclared identifier 'foo'" in the example above.
  std::string message;
  /// Contains the fully rendered error message, without "error: ",
  /// but including the source context.
  std::string rendered;
};

StructuredData::ObjectSP Serialize(llvm::ArrayRef<DiagnosticDetail> details);

void RenderDiagnosticDetails(Stream &stream,
                             std::optional<uint16_t> offset_in_command,
                             bool show_inline,
                             llvm::ArrayRef<DiagnosticDetail> details);

class DiagnosticError
    : public llvm::ErrorInfo<DiagnosticError, CloneableECError> {
public:
  using llvm::ErrorInfo<DiagnosticError, CloneableECError>::ErrorInfo;
  DiagnosticError(std::error_code ec) : ErrorInfo(ec) {}
  lldb::ErrorType GetErrorType() const override;
  virtual llvm::ArrayRef<DiagnosticDetail> GetDetails() const = 0;
  StructuredData::ObjectSP GetAsStructuredData() const override {
    return Serialize(GetDetails());
  }
  static char ID;
};

} // namespace lldb_private
#endif
