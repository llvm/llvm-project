// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -ferror-limit 1 %s 2>&1 | FileCheck %s
unknown_type foo(unknown_type);
// CHECK: fatal error: too many errors emitted, stopping now

template <typename>
class Bar {};

extern template class Bar<int>;
template class Bar<int>;
