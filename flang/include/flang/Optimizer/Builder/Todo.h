//===-- Optimizer/Builder/Todo.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_TODO_H
#define FORTRAN_LOWER_TODO_H

#include "flang/Optimizer/Support/FatalError.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

// This is throw-away code used to mark areas of the code that have not yet been
// developed.

#undef TODO
// Use TODO_NOLOC if no mlir location is available to indicate the line in
// Fortran source file that requires an unimplemented feature.
#undef TODO_NOLOC

#undef TODOQUOTE
#define TODOQUOTE(X) #X

// Give backtrace only in debug builds.
#undef GEN_TRACE
#ifdef NDEBUG
#define GEN_TRACE false
#else
#define GEN_TRACE true
#endif

#undef TODO_NOLOCDEFN
#define TODO_NOLOCDEFN(ToDoMsg, ToDoFile, ToDoLine, GenTrace)                  \
  do {                                                                         \
    llvm::report_fatal_error(llvm::Twine(ToDoFile ":" TODOQUOTE(               \
                                 ToDoLine) ": not yet implemented: ") +        \
                                 llvm::Twine(ToDoMsg),                         \
                             GenTrace);                                        \
  } while (false)

#define TODO_NOLOC(ToDoMsg) TODO_NOLOCDEFN(ToDoMsg, __FILE__, __LINE__, false)
#define TODO_NOLOC_TRACE(ToDoMsg)                                              \
  TODO_NOLOCDEFN(ToDoMsg, __FILE__, __LINE__, GENTRACE)

#undef TODO_DEFN
#define TODO_DEFN(MlirLoc, ToDoMsg, ToDoFile, ToDoLine, GenTrace)              \
  do {                                                                         \
    fir::emitFatalError(MlirLoc,                                               \
                        llvm::Twine(ToDoFile ":" TODOQUOTE(                    \
                            ToDoLine) ": not yet implemented: ") +             \
                            llvm::Twine(ToDoMsg),                              \
                        GenTrace);                                             \
  } while (false)

#define TODO(MlirLoc, ToDoMsg)                                                 \
  TODO_DEFN(MlirLoc, ToDoMsg, __FILE__, __LINE__, false)
#define TODO_TRACE(MlirLoc, ToDoMsg)                                           \
  TODO_DEFN(MlirLoc, ToDoMsg, __FILE__, __LINE__, GEN_TRACE)

#endif // FORTRAN_LOWER_TODO_H
