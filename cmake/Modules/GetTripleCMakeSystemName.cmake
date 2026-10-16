#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Map a target triple to the corresponding CMake system name.
#
# Usage:
#   get_triple_cmake_system_name(<triple> <out_var>)
#
# Sets <out_var> to the CMake-style system name
# (e.g. "x86_64-pc-linux-gnu" -> Linux, "arm64-apple-macos" ->
# "Darwin"). A triple with an OS cmake does not recognize maps to
# "Generic".
#
# The OS/environment -> system name data is derived from
# llvm/include/llvm/TargetParser/TripleName.def

get_filename_component(_gtcsn_llvm_dir "${CMAKE_CURRENT_LIST_DIR}/../../llvm"
                       ABSOLUTE)
set(_gtcsn_script "${_gtcsn_llvm_dir}/utils/get_triple_system_name.py")

function(get_triple_cmake_system_name triple out_var)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_gtcsn_script}" "--triple" "${triple}"
    OUTPUT_VARIABLE _name
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _result)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Failed to derive CMake system name for '${triple}'")
  endif()
  if(_name)
    set(${out_var} "${_name}" PARENT_SCOPE)
  else()
    set(${out_var} "${CMAKE_HOST_SYSTEM_NAME}" PARENT_SCOPE)
  endif()
endfunction()
