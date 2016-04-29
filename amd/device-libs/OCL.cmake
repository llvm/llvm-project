################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
## 
## Copyright (c) 2016, Advanced Micro Devices, Inc. All rights reserved.
## 
## Developed by:
## 
##                 AMD Research and AMD HSA Software Development
## 
##                 Advanced Micro Devices, Inc.
## 
##                 www.amd.com
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and#or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
## 
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and#or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

set (LLVM_LINK "${LLVM_TOOLS_BINARY_DIR}/llvm-link")
set (LLVM_OBJDUMP "${LLVM_TOOLS_BINARY_DIR}/llvm-objdump")
set (CMAKE_OCL_OUTPUT_EXTENTION .bc)
set (CMAKE_OCL_OUTPUT_EXTENTION_REPLACE 1)
set (CMAKE_OCL_COMPILER ${CMAKE_C_COMPILER})
set (CMAKE_OCL_COMPILE_OBJECT "<CMAKE_OCL_COMPILER> -o <OBJECT> <FLAGS> -c <SOURCE>")
set (CMAKE_OCL_LINK_EXECUTABLE "${LLVM_LINK} <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
set (CMAKE_OCL_CREATE_STATIC_LIBRARY "${LLVM_LINK} -o <TARGET> <LINK_FLAGS> <OBJECTS>")

macro(clang_opencl_bc_lib name files)
  add_library(${name}_lib_bc STATIC ${files})
  set_target_properties(${name}_lib_bc PROPERTIES OUTPUT_NAME ${name})
  set_target_properties(${name}_lib_bc PROPERTIES PREFIX "" SUFFIX ".lib.bc")
  set_target_properties(${name}_lib_bc PROPERTIES COMPILE_FLAGS "-x cl -target amdgcn--amdhsa -emit-llvm")
  set_target_properties(${name}_lib_bc PROPERTIES LINKER_LANGUAGE OCL)
endmacro(clang_opencl_bc_lib)

macro(clang_opencl_bc_exe name)
  add_executable(${name}_exe_bc "")
  foreach (lib ${ARGN})
    target_link_libraries(${name}_exe_bc $<TARGET_FILE:${lib}>)
    add_dependencies(${name}_exe_bc ${lib})
  endforeach()
  set_target_properties(${name}_exe_bc PROPERTIES LINKER_LANGUAGE OCL)
  set_target_properties(${name}_exe_bc PROPERTIES OUTPUT_NAME ${name})
  set_target_properties(${name}_exe_bc PROPERTIES PREFIX "" SUFFIX ".exe.bc")
  set_target_properties(${name}_exe_bc PROPERTIES LINKER_LANGUAGE OCL)
endmacro(clang_opencl_bc_exe)

macro(clang_opencl_bc_all_exe name)
  clang_opencl_bc_lib(${name} ${name}.c)
  clang_opencl_bc_exe(${name} ${name}_lib_bc opencl_amdgpu_lib_bc)
endmacro(clang_opencl_bc_all_exe)

macro(clang_opencl_code name)
  clang_opencl_bc_all_exe(${name})
  add_executable(${name}_code "")
  target_link_libraries(${name}_code $<TARGET_FILE:${name}_exe_bc>)
  set_target_properties(${name}_code PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(${name}_code PROPERTIES OUTPUT_NAME ${name})
  set_target_properties(${name}_code PROPERTIES LINK_FLAGS "-target amdgcn--amdhsa")
  set_target_properties(${name}_code PROPERTIES PREFIX "" SUFFIX ".co")
  add_dependencies(${name}_code ${name}_exe_bc)
  install (TARGETS ${name}_code DESTINATION test COMPONENT OpenCL-Lib-test)
endmacro(clang_opencl_code)

enable_testing()

macro(clang_opencl_test name)
  clang_opencl_code(${name})
  if(AMDHSACOD)
    add_test(
      NAME ${name}:llvm-objdump
      COMMAND ${LLVM_OBJDUMP} -disassemble-all $<TARGET_FILE:${name}_code>
    )
    add_test(
      NAME ${name}:amdhsacod
      COMMAND ${AMDHSACOD} -test -code $<TARGET_FILE:${name}_code>
    )
  endif()
endmacro(clang_opencl_test)
