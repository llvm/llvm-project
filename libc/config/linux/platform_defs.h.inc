//===----- Definition of platform specific macros for ELF paltforms -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

%%begin()

#define ENTRYPOINT_SECTION_ATTRIBUTE(name) \
    __attribute__((section(".llvm.libc.entrypoint."#name)))
#define LLVM_LIBC_ENTRYPOINT(name) ENTRYPOINT_SECTION_ATTRIBUTE(name) name
