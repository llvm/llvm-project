// RUN: cat %S/Inputs/modernize-pass-by-value/header-with-fix.h > %T/pass-by-value-header-with-fix.h
// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,modernize-pass-by-value' -header-filter='.*' -fix -- -std=c++11 -I %T | FileCheck %s -check-prefix=CHECK-MESSAGES -implicit-check-not="{{warning|error}}:"
// RUN: FileCheck -input-file=%t.cpp %s -check-prefix=CHECK-FIXES
// RUN: FileCheck -input-file=%T/pass-by-value-header-with-fix.h %s -check-prefix=CHECK-HEADER-FIXES

#include "pass-by-value-header-with-fix.h"
// CHECK-HEADER-FIXES: Foo(S s);
Foo::Foo(const S &s) : s(s) {}
// CHECK-MESSAGES: :9:10: warning: pass by value and use std::move [modernize-pass-by-value]
// CHECK-FIXES: #include <utility>
// CHECK-FIXES: Foo::Foo(S s) : s(std::move(s)) {}
