#===-- CMakeLists.txt ----------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#

add_object_library(
  pwd_utils
  SRCS
    pwd_utils.cpp
  HDRS
    pwd_utils.h
  DEPENDS
    libc.hdr.types.struct_passwd
    libc.src.__support.str_to_integer
    libc.src.string.string_utils
)

add_entrypoint_object(
  getpwent
  SRCS
    getpwent.cpp
  HDRS
    getpwent.h
  DEPENDS
    libc.include.pwd
    libc.hdr.types.struct_passwd
    libc.src.errno.errno
    libc.src.stdio.fopen
    libc.src.stdio.fclose
    libc.src.stdio.fgets
    libc.src.stdio.fseek
    libc.src.string.string_utils
    libc.hdr.stdio_macros
    .pwd_utils
)

add_entrypoint_object(
  setpwent
  SRCS
    setpwent.cpp
  HDRS
    setpwent.h
  DEPENDS
    libc.include.pwd
    .getpwent
    .pwd_utils
)

add_entrypoint_object(
  endpwent
  SRCS
    endpwent.cpp
  HDRS
    endpwent.h
  DEPENDS
    libc.include.pwd
    .getpwent
    .pwd_utils
)
