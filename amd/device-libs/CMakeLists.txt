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

cmake_minimum_required(VERSION 2.6)

project(ROCm-OpenCL-Lib)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
endif()

if (AMDHSACOD)
else()
  message(WARNING "amdhsacod is not found, testing disabled")
  message(WARNING "Specify amdhsacod executable with -DAMDHSACOD=<...>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} "/opt/rocm/llvm" NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

include_directories(${CMAKE_SOURCE_DIR}/inc)

list(APPEND CMAKE_OCL_SOURCE_FILE_EXTENSIONS .cl)
list(APPEND CMAKE_C_SOURCE_FILE_EXTENSIONS .cl)
list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS .cl)

set (CLANG_OCL_FLAGS "-x cl -Xclang -cl-std=CL2.0 -fblocks -target amdgcn--amdhsa -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200 -Dcl_khr_fp64 -Dcl_khr_fp16 -include opencl.h")
set (CLANG_OCL_LINK_FLAGS "-target amdgcn--amdhsa -mcpu=fiji")

include(OCL.cmake)

add_subdirectory(utils/prepare-builtins)
add_subdirectory(llvm)
add_subdirectory(ocml)
add_subdirectory(opencl)
add_subdirectory(test/opencl)
