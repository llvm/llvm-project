//===- Diagnostics.h - Helpers for diagnostics in Python bindings ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_DIAGNOSTICS_H
#define AIIR_BINDINGS_PYTHON_DIAGNOSTICS_H

#include "aiir-c/Diagnostics.h"
#include "aiir-c/IR.h"

#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>

namespace aiir {
namespace python {

/// RAII scope intercepting all diagnostics into a string. The message must be
/// checked before this goes out of scope.
class CollectDiagnosticsToStringScope {
public:
  explicit CollectDiagnosticsToStringScope(AiirContext ctx) : context(ctx) {
    handlerID =
        aiirContextAttachDiagnosticHandler(ctx, &handler, &messageStream,
                                           /*deleteUserData=*/nullptr);
  }
  ~CollectDiagnosticsToStringScope() {
    assert(messageStream.str().empty() && "unchecked error message");
    aiirContextDetachDiagnosticHandler(context, handlerID);
  }

  [[nodiscard]] std::string takeMessage() {
    std::string newMessage = messageStream.str();
    messageStream.str("");
    messageStream.clear();
    return newMessage;
  }

private:
  static AiirLogicalResult handler(AiirDiagnostic diag, void *data) {
    auto printer = +[](AiirStringRef message, void *data) {
      *static_cast<std::ostringstream *>(data)
          << std::string_view(message.data, message.length);
    };
    AiirLocation loc = aiirDiagnosticGetLocation(diag);
    *static_cast<std::ostringstream *>(data) << "at ";
    aiirLocationPrint(loc, printer, data);
    *static_cast<std::ostringstream *>(data) << ": ";
    aiirDiagnosticPrint(diag, printer, data);
    for (intptr_t i = 0; i < aiirDiagnosticGetNumNotes(diag); i++) {
      *static_cast<std::ostringstream *>(data) << "\n";
      AiirDiagnostic note = aiirDiagnosticGetNote(diag, i);
      handler(note, data);
    }
    return aiirLogicalResultSuccess();
  }

  AiirContext context;
  AiirDiagnosticHandlerID handlerID;

  std::ostringstream messageStream;
};

} // namespace python
} // namespace aiir

#endif // AIIR_BINDINGS_PYTHON_DIAGNOSTICS_H
