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

// TODO: Get rid of the PAGE_SIZE macro. It is present only as an interim
// measure until we can move the implementations of mmap and munmap to under
// the config/linux directory. After that, the implementations can use
// EXEC_PAGESIZE until page size can be read from the aux vector. 
#define PAGE_SIZE 4096
