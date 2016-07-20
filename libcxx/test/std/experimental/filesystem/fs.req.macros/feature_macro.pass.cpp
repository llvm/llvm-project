//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// #define __cpp_lib_experimental_filesystem 201406L

#include <experimental/filesystem>

#ifndef __cpp_lib_experimental_filesystem
#error Filesystem feature test macro is not defined  (__cpp_lib_experimental_filesystem)
#elif __cpp_lib_experimental_filesystem != 201406L
#error Filesystem feature test macro has an incorrect value (__cpp_lib_experimental_filesystem)
#endif

int main() { }
