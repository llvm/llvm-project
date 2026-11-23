// From https://github.com/llvm/llvm-project/issues/245313
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Build two different BMIs for module a. b is built against the first one,
// while the final translation unit uses the second one. The decl IDs recorded
// in b no longer match the decl table of a, which used to crash clang while
// deserializing declaration update records.
//
// RUN: %clang -std=c++20 -x c++-module -fmodule-output=%t/a.pcm -o /dev/null -c %t/a.cpp
// RUN: %clang -std=c++20 -x c++-module -fmodule-output=%t/a.bmi -o %t/a.bmi --precompile %t/a.cpp
// RUN: %clang -std=c++20 -x c++-module -fmodule-output=%t/b.pcm -fmodule-file=a=%t/a.bmi -o /dev/null -c %t/b.cpp
// RUN: not %clang -std=c++20 -fmodule-file=b=%t/b.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only %t/c.cpp 2>&1 | FileCheck %s
//
// CHECK: declaration ID out-of-range for AST file

//--- a.cpp
module;
namespace std {}
export module a;

//--- b.cpp
export module b;
import a;

//--- c.cpp
import b;
import a;
