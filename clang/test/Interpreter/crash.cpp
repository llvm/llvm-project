// REQUIRES: host-supports-jit, x86_64-linux

// RUN: rm -rf %t
// RUN: mkdir -p %t
//
// RUN: split-file %s %t
//
// RUN: %clang++ -std=c++20 -fPIC -c %t/vec.cpp -o %t/vec.o
// RUN: %clang++ -shared %t/vec.o -o %t/vec.so
//
// RUN: cat %t/Test.cpp | LD_LIBRARY_PATH=%t:$LD_LIBRARY_PATH clang-repl

//--- vec.cpp
#include <vector>

//--- Test.cpp
%lib vec.so
#include <vector>
std::vector<int> v;
%quit
