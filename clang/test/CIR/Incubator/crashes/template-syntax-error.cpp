// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir 2>&1 | FileCheck %s
//
// This test captures malformed C++ that creduce generated.
// It documents that our crash reproducer reduction process needs improvement.
//
// CHECK: error: expected
//
// Issue: Creduce produced syntactically invalid C++ during reduction
//
// This is a creduce artifact showing incomplete template syntax.
// The original crash involved template metaprogramming, but creduce
// reduced it too aggressively, producing invalid syntax.

template <a> b() struct c {
  c::b::
