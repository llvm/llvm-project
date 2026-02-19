// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -Xclang -emit-pch -Xclang -include -Xclang %t/cmake_pch.hxx -MD -MT %t/cmake_pch.hxx.pch -MF %t/cmake_pch.hxx.pch.d -o %t/cmake_pch.hxx.pch -c %t/cmake_pch.hxx.cxx

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -Xclang -include-pch -Xclang %t/cmake_pch.hxx.pch -Xclang -include -Xclang %t/cmake_pch.hxx -MD -MT %t/Test.obj -MF %t/Test.obj.d -o %t/Test.obj -c %t/Test.cpp

//--- Test.cpp
#include "pch.h"

int foo() {
  return 42;
}

//--- pch.h
#pragma once

//--- cmake_pch.hxx.cxx
// empty

//--- cmake_pch.hxx
#pragma clang system_header
#include "pch.h"
