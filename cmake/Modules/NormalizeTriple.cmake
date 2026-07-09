#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Normalize a target triple using clang's -print-target-triple.
#
# Usage:
#   normalize_triple(<compiler> <triple> <out_var>)
#
# Runs <compiler> --target=<triple> -print-target-triple to produce a
# canonical triple. If the compiler invocation fails (e.g. the compiler
# is not clang), <triple> is returned unchanged.

function(normalize_triple compiler triple out_var)
  set(_prefix "")
  if(CMAKE_C_COMPILER_FRONTEND_VARIANT MATCHES "MSVC")
    set(_prefix "/clang:")
  endif()
  execute_process(
    COMMAND "${compiler}" "${_prefix}--target=${triple}" "${_prefix}-print-target-triple"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_result EQUAL 0 AND _output)
    set(${out_var} "${_output}" PARENT_SCOPE)
  else()
    # TODO(#97876): Report an error.
    message(WARNING "Failed to execute `${compiler} ${_prefix}--target=${triple} ${_prefix}-print-target-triple` to normalize target triple.")
    set(${out_var} "${triple}" PARENT_SCOPE)
  endif()
endfunction()
