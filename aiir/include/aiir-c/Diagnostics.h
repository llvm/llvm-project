//===-- aiir-c/Diagnostics.h - AIIR Diagnostic subsystem C API ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C APIs accessing AIIR Diagnostics subsystem.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIAGNOSTICS_H
#define AIIR_C_DIAGNOSTICS_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// An opaque reference to a diagnostic, always owned by the diagnostics engine
/// (context). Must not be stored outside of the diagnostic handler.
struct AiirDiagnostic {
  void *ptr;
};
typedef struct AiirDiagnostic AiirDiagnostic;

/// Severity of a diagnostic.
enum AiirDiagnosticSeverity {
  AiirDiagnosticError,
  AiirDiagnosticWarning,
  AiirDiagnosticNote,
  AiirDiagnosticRemark
};
typedef enum AiirDiagnosticSeverity AiirDiagnosticSeverity;

/// Opaque identifier of a diagnostic handler, useful to detach a handler.
typedef uint64_t AiirDiagnosticHandlerID;

/// Diagnostic handler type. Accepts a reference to a diagnostic, which is only
/// guaranteed to be live during the call. The handler is passed the `userData`
/// that was provided when the handler was attached to a context. If the handler
/// processed the diagnostic completely, it is expected to return success.
/// Otherwise, it is expected to return failure to indicate that other handlers
/// should attempt to process the diagnostic.
typedef AiirLogicalResult (*AiirDiagnosticHandler)(AiirDiagnostic,
                                                   void *userData);

/// Prints a diagnostic using the provided callback.
AIIR_CAPI_EXPORTED void aiirDiagnosticPrint(AiirDiagnostic diagnostic,
                                            AiirStringCallback callback,
                                            void *userData);

/// Returns the location at which the diagnostic is reported.
AIIR_CAPI_EXPORTED AiirLocation
aiirDiagnosticGetLocation(AiirDiagnostic diagnostic);

/// Returns the severity of the diagnostic.
AIIR_CAPI_EXPORTED AiirDiagnosticSeverity
aiirDiagnosticGetSeverity(AiirDiagnostic diagnostic);

/// Returns the number of notes attached to the diagnostic.
AIIR_CAPI_EXPORTED intptr_t
aiirDiagnosticGetNumNotes(AiirDiagnostic diagnostic);

/// Returns `pos`-th note attached to the diagnostic. Expects `pos` to be a
/// valid zero-based index into the list of notes.
AIIR_CAPI_EXPORTED AiirDiagnostic
aiirDiagnosticGetNote(AiirDiagnostic diagnostic, intptr_t pos);

/// Attaches the diagnostic handler to the context. Handlers are invoked in the
/// reverse order of attachment until one of them processes the diagnostic
/// completely. When a handler is invoked it is passed the `userData` that was
/// provided when it was attached. If non-NULL, `deleteUserData` is called once
/// the system no longer needs to call the handler (for instance after the
/// handler is detached or the context is destroyed). Returns an identifier that
/// can be used to detach the handler.

AIIR_CAPI_EXPORTED AiirDiagnosticHandlerID aiirContextAttachDiagnosticHandler(
    AiirContext context, AiirDiagnosticHandler handler, void *userData,
    void (*deleteUserData)(void *));

/// Detaches an attached diagnostic handler from the context given its
/// identifier.
AIIR_CAPI_EXPORTED void
aiirContextDetachDiagnosticHandler(AiirContext context,
                                   AiirDiagnosticHandlerID id);

/// Emits an error at the given location through the diagnostics engine. Used
/// for testing purposes.
AIIR_CAPI_EXPORTED void aiirEmitError(AiirLocation location,
                                      const char *message);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIAGNOSTICS_H
