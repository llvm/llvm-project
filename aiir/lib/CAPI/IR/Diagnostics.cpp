//===- Diagnostics.cpp - C Interface for AIIR Diagnostics -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Diagnostics.h"
#include "aiir/CAPI/Diagnostics.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/CAPI/Utils.h"
#include "aiir/IR/Diagnostics.h"

using namespace aiir;

void aiirDiagnosticPrint(AiirDiagnostic diagnostic, AiirStringCallback callback,
                         void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(diagnostic).print(stream);
}

AiirLocation aiirDiagnosticGetLocation(AiirDiagnostic diagnostic) {
  return wrap(unwrap(diagnostic).getLocation());
}

AiirDiagnosticSeverity aiirDiagnosticGetSeverity(AiirDiagnostic diagnostic) {
  switch (unwrap(diagnostic).getSeverity()) {
  case aiir::DiagnosticSeverity::Error:
    return AiirDiagnosticError;
  case aiir::DiagnosticSeverity::Warning:
    return AiirDiagnosticWarning;
  case aiir::DiagnosticSeverity::Note:
    return AiirDiagnosticNote;
  case aiir::DiagnosticSeverity::Remark:
    return AiirDiagnosticRemark;
  }
  llvm_unreachable("unhandled diagnostic severity");
}

// Notes are stored in a vector, so note iterator range is a pair of
// random access iterators, for which it is cheap to compute the size.
intptr_t aiirDiagnosticGetNumNotes(AiirDiagnostic diagnostic) {
  return static_cast<intptr_t>(llvm::size(unwrap(diagnostic).getNotes()));
}

// Notes are stored in a vector, so the iterator is a random access iterator,
// cheap to advance multiple steps at a time.
AiirDiagnostic aiirDiagnosticGetNote(AiirDiagnostic diagnostic, intptr_t pos) {
  return wrap(*std::next(unwrap(diagnostic).getNotes().begin(), pos));
}

static void deleteUserDataNoop(void *userData) {}

AiirDiagnosticHandlerID aiirContextAttachDiagnosticHandler(
    AiirContext context, AiirDiagnosticHandler handler, void *userData,
    void (*deleteUserData)(void *)) {
  assert(handler && "unexpected null diagnostic handler");
  if (deleteUserData == nullptr)
    deleteUserData = deleteUserDataNoop;
  DiagnosticEngine::HandlerID id =
      unwrap(context)->getDiagEngine().registerHandler(
          [handler,
           ownedUserData = std::unique_ptr<void, decltype(deleteUserData)>(
               userData, deleteUserData)](Diagnostic &diagnostic) {
            return unwrap(handler(wrap(diagnostic), ownedUserData.get()));
          });
  return static_cast<AiirDiagnosticHandlerID>(id);
}

void aiirContextDetachDiagnosticHandler(AiirContext context,
                                        AiirDiagnosticHandlerID id) {
  unwrap(context)->getDiagEngine().eraseHandler(
      static_cast<DiagnosticEngine::HandlerID>(id));
}

void aiirEmitError(AiirLocation location, const char *message) {
  emitError(unwrap(location)) << message;
}
