/*==-- clang-c/Driver.h - A C Interface for the Clang Driver ------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a C API for extracting information from the clang     *|
|* driver.                                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifndef CLANG_CLANG_C_DRIVER
#define CLANG_CLANG_C_DRIVER

#include "clang-c/Index.h"
#include "clang-c/Platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Contains the command line arguments for an external action.  Same format as
 * provided to main.
 */
typedef struct {
  /* Number of arguments in ArgV */
  int ArgC;
  /* Null terminated array of pointers to null terminated argument strings */
  const char **ArgV;
} CXExternalAction;

/**
 * Contains the list of external actions clang would invoke.
 */
typedef struct {
  int Count;
  CXExternalAction **Actions;
} CXExternalActionList;

/**
 * Get the external actions that the clang driver will invoke for the given
 * command line.
 *
 * \param ArgC number of arguments in \p ArgV.
 * \param ArgV array of null terminated arguments.  Doesn't need to be null
 *   terminated.
 * \param Environment must be null.
 * \param WorkingDirectory a null terminated path to the working directory to
 *   use for this invocation.  `nullptr` to use the current working directory of
 *   the process.
 * \param OutDiags will be set to a \c CXDiagnosticSet if there's an error.
 *   Must be freed by calling \c clang_disposeDiagnosticSet .
 * \returns A pointer to a \c CXExternalActionList on success, null on failure.
 *   The returned \c CXExternalActionList must be freed by calling
 *   \c clang_Driver_ExternalActionList_dispose .
 */
CINDEX_LINKAGE CXExternalActionList *
clang_Driver_getExternalActionsForCommand_v0(int ArgC, const char **ArgV,
                                             const char **Environment,
                                             const char *WorkingDirectory,
                                             CXDiagnosticSet *OutDiags);

/**
 * Deallocate a \c CXExternalActionList
 */
CINDEX_LINKAGE void
clang_Driver_ExternalActionList_dispose(CXExternalActionList *EAL);

#ifdef __cplusplus
}
#endif

#endif // CLANG_CLANG_C_DRIVER
