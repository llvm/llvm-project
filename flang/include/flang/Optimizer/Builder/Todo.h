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

#ifdef NDEBUG

// In a release build, just give a message and exit.
#define TODO_NOLOC(ToDoMsg)                                                    \
  do {                                                                         \
    llvm::errs() << __FILE__ << ':' << __LINE__                                \
                 << ": not yet implemented: " << ToDoMsg << '\n';              \
    std::exit(1);                                                              \
  } while (false)

#undef TODO_DEFN
#define TODO_DEFN(MlirLoc, ToDoMsg, ToDoFile, ToDoLine)                        \
  do {                                                                         \
    mlir::emitError(MlirLoc, ToDoFile ":" TODOQUOTE(                           \
                                 ToDoLine) ": not yet implemented: " ToDoMsg); \
    std::exit(1);                                                              \
  } while (false)

#define TODO(MlirLoc, ToDoMsg) TODO_DEFN(MlirLoc, ToDoMsg, __FILE__, __LINE__)

#else

// In a developer build, print a message and give a backtrace.
#undef TODO_NOLOCDEFN
#define TODO_NOLOCDEFN(ToDoMsg, ToDoFile, ToDoLine)                            \
  do {                                                                         \
    llvm::report_fatal_error(                                                  \
        ToDoFile ":" TODOQUOTE(ToDoLine) ": not yet implemented: " ToDoMsg);   \
  } while (false)

#define TODO_NOLOC(ToDoMsg) TODO_NOLOCDEFN(ToDoMsg, __FILE__, __LINE__)

#undef TODO_DEFN
#define TODO_DEFN(MlirLoc, ToDoMsg, ToDoFile, ToDoLine)                        \
  do {                                                                         \
    fir::emitFatalError(                                                       \
        MlirLoc,                                                               \
        ToDoFile ":" TODOQUOTE(ToDoLine) ": not yet implemented: " ToDoMsg);   \
  } while (false)

#define TODO(MlirLoc, ToDoMsg) TODO_DEFN(MlirLoc, ToDoMsg, __FILE__, __LINE__)

#endif

#endif // FORTRAN_LOWER_TODO_H
