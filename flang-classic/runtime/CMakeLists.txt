#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set (RUNTIME_SHARED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shared)

add_definitions(
 -DMAXCPUS=256 
 -DMAXCPUSL=8 
 -DMAXCPUSR=8
 -DTARGET_LLVM
 -DPGF90 
 -DPGFLANG
 -DNATIVE_FPCVT
 -DPGI_LITTLE_ENDIAN
 -DINLINE_MEMOPS
 -DTARGET_${OS}
 -D${OS}
)

if( ${TARGET_ARCHITECTURE} STREQUAL "x86_64" )
  add_definitions(
   -DTARGET_X8664
   -DTARGET_${OS}_X8664
  )
elseif( ${TARGET_ARCHITECTURE} STREQUAL "aarch64" )
  add_definitions(
   -DTARGET_LLVM_ARM64
   -DTARGET_${OS}_ARM
   -DTARGET_${OS}_ARM64
  )
elseif( ${TARGET_ARCHITECTURE} STREQUAL "ppc64le" )
  add_definitions(
    -DTARGET_${OS}_POWER
    -D${OS}_POWER
  )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(ompstub)

add_subdirectory(flangrti)
add_subdirectory(flang)
add_subdirectory(flangmain)
