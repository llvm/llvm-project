#===-- cmake/modules/HandleLibs.cmake --------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Select the C library to use for building flang-rt.
if (FLANG_RT_LIBC_PROVIDER STREQUAL "system")
  add_library(flang-rt-libc-headers INTERFACE)
  add_library(flang-rt-libc-static INTERFACE)
  add_library(flang-rt-libc-shared INTERFACE)
elseif (FLANG_RT_LIBC_PROVIDER STREQUAL "llvm")
  add_library(flang-rt-libc-headers INTERFACE)
  target_link_libraries(flang-rt-libc-headers INTERFACE libc-headers)
  if (FLANG_RT_HAS_NOSTDLIBINC_FLAG)
    target_compile_options(flang-rt-libc-headers INTERFACE $<$<COMPILE_LANGUAGE:CXX,C>:-nostdlibinc>)
  endif ()

  add_library(flang-rt-libc-static INTERFACE)
  if (TARGET libc)
    target_link_libraries(flang-rt-libc-static INTERFACE libc)
  endif ()
  if (TARGET libm)
    target_link_libraries(flang-rt-libc-static INTERFACE libm)
  endif ()
  if (FLANG_RT_HAS_NOSTDLIB_FLAG)
    target_compile_options(flang-rt-libc-headers INTERFACE $<$<COMPILE_LANGUAGE:CXX,C>:-nostdlib>)
  endif ()

  # TODO: There's no support for building LLVM libc as a shared library yet.
  add_library(flang-rt-libc-shared INTERFACE)
endif ()

# Select the C++ library to use for building flang-rt.
if (FLANG_RT_LIBCXX_PROVIDER STREQUAL "system")
  add_library(flang-rt-libcxx-headers INTERFACE)
elseif (FLANG_RT_LIBCXX_PROVIDER STREQUAL "llvm")
  add_library(flang-rt-libcxx-headers INTERFACE)
  target_link_libraries(flang-rt-libcxx-headers INTERFACE cxx-headers)

  if (CXX_SUPPORTS_NOSTDINCXX_FLAG)
    target_compile_options(flang-rt-libc-headers INTERFACE $<$<COMPILE_LANGUAGE:CXX,C>:-nostdinc++>)
  endif ()
endif ()
