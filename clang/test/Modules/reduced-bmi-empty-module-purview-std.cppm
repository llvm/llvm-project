// Test that we won't write additional information from std namespace by default
// into the Reduced BMI if the module purview is empty.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram --show-binary-blobs %t/A.pcm > %t/A.dump
// RUN: cat %t/A.dump | FileCheck %t/A.cppm

//--- std.h
namespace std {
  typedef decltype(sizeof(0)) size_t;
  enum class align_val_t : std::size_t {};

  class bad_alloc { };
}

//--- A.cppm
module;
#include "std.h"
export module A;

// CHECK-NOT: <DECL_NAMESPACE
// CHECK-NOT: <DECL_CONTEXT_LEXICAL
// CHECK-NOT: <DELAYED_NAMESPACE_LEXICAL_VISIBLE_RECORD
