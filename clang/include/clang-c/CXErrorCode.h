/*===-- clang-c/CXErrorCode.h - C Index Error Codes  --------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides the CXErrorCode enumerators.                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_CLANG_C_CXERRORCODE_H
#define LLVM_CLANG_C_CXERRORCODE_H

#include "clang-c/Platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes returned by libclang routines.
 *
 * Zero (\c CXError_Success) is the only error code indicating success.  Other
 * error codes, including not yet assigned non-zero values, indicate errors.
 */
enum CXErrorCode {
  /**
   * No error.
   */
  CXError_Success = 0,

  /**
   * A generic error code, no further details are available.
   *
   * Errors of this kind can get their own specific error codes in future
   * libclang versions.
   */
  CXError_Failure = 1,

  /**
   * libclang crashed while performing the requested operation.
   */
  CXError_Crashed = 2,

  /**
   * The function detected that the arguments violate the function
   * contract.
   */
  CXError_InvalidArguments = 3,

  /**
   * An AST deserialization error has occurred.
   */
  CXError_ASTReadError = 4,

  /**
  * \brief A refactoring action is not available at the given location
  * or in the given source range.
  */
  CXError_RefactoringActionUnavailable = 5,

  /**
  * \brief A refactoring action is not able to use the given name because
  * it contains an unexpected number of strings.
  */
  CXError_RefactoringNameSizeMismatch = 6,

  /**
  * \brief A name of a symbol is invalid, i.e. it is reserved by the source
  * language and can't be used as a name for this symbol.
  */
  CXError_RefactoringNameInvalid = 7
};

#ifdef __cplusplus
}
#endif
#endif

