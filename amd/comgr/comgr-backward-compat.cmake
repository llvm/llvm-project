# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.16.8)

set(COMGR_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(COMGR_WRAPPER_DIR ${COMGR_BUILD_DIR}/wrapper_dir)
set(COMGR_WRAPPER_INC_DIR ${COMGR_WRAPPER_DIR}/include)

#Function to generate header template file
function(create_header_template)
    file(WRITE ${COMGR_WRAPPER_DIR}/header.hpp.in "/*
    Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the \"Software\"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */

#ifndef @include_guard@
#define @include_guard@

#ifndef ROCM_HEADER_WRAPPER_WERROR
#define ROCM_HEADER_WRAPPER_WERROR @deprecated_error@
#endif
#if ROCM_HEADER_WRAPPER_WERROR  /* ROCM_HEADER_WRAPPER_WERROR 1 */
#error \"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with amd_comgr\"
#else   /* ROCM_HEADER_WRAPPER_WERROR 0 */
#if defined(__GNUC__)
#warning \"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with amd_comgr\"
#else
#pragma message(\"This file is deprecated. Use file from include path /opt/rocm-ver/include/ and prefix with amd_comgr\")
#endif
#endif /* ROCM_HEADER_WRAPPER_WERROR */

@include_statements@

#endif")
endfunction()

#use header template file and generate wrapper header files
function(generate_wrapper_header)
  file(MAKE_DIRECTORY ${COMGR_WRAPPER_INC_DIR})
  #find all header files(*.h) from include
  file(GLOB include_files ${COMGR_BUILD_DIR}/include/*.h)
  #Generate wrapper header files for each files in the list
  foreach(header_file ${include_files})
    # set include guard
    get_filename_component(INC_GAURD_NAME ${header_file} NAME_WE)
    string(TOUPPER ${INC_GAURD_NAME} INC_GAURD_NAME)
    set(include_guard "${include_guard}COMGR_WRAPPER_INCLUDE_${INC_GAURD_NAME}_H")
    #set #include statement
    get_filename_component(file_name ${header_file} NAME)
    set(include_statements "${include_statements}#include \"${amd_comgr_NAME}/${file_name}\"\n")
    configure_file(${COMGR_WRAPPER_DIR}/header.hpp.in ${COMGR_WRAPPER_INC_DIR}/${file_name})
    unset(include_statements)
    unset(include_guard)
  endforeach()

endfunction()

#Creater a template for header file
create_header_template()
#Use template header file and generater wrapper header files
generate_wrapper_header()
install(DIRECTORY ${COMGR_WRAPPER_INC_DIR} COMPONENT amd-comgr DESTINATION .)
