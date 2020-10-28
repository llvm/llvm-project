//===-- Lower/Todo.h --------------------------------------------*- C++ -*-===//
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

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

// This is throw-away code used to mark areas of the code that have not yet been
// developed.

#undef TODO

#ifdef NDEBUG

// In a release build, just give a message and exit.
#define TODO(ToDoMsg)                                                          \
  do {                                                                         \
    llvm::errs() << __FILE__ << ':' << __LINE__ << ": not yet implemented "    \
                 << ToDoMsg << '\n';                                           \
    std::exit(1);                                                              \
  } while (false)

#else

#undef TODOQUOTE
#define TODOQUOTE(X) #X

// In a developer build, print a message and give a backtrace.
#define TODO(ToDoMsg)                                                          \
  do {                                                                         \
    llvm::report_fatal_error(                                                  \
        __FILE__ ":" TODOQUOTE(__LINE__) ": not yet implemented " ToDoMsg);    \
  } while (false)

#endif

#endif // FORTRAN_LOWER_TODO_H
