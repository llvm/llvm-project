// RUN: cp %S/Inputs/modernize-pass-by-value/header.h %T/pass-by-value-header.h
// RUN: clang-tidy %s -checks='-*,modernize-pass-by-value' -header-filter='.*' -fix -- -std=c++11 -I %T | FileCheck %s -check-prefix=CHECK-MESSAGES -implicit-check-not="{{warning|error}}:"
// RUN: FileCheck -input-file=%T/pass-by-value-header.h %s -check-prefix=CHECK-FIXES
// FIXME: Make the test work in all language modes.

#include "pass-by-value-header.h"
// CHECK-MESSAGES: :8:5: warning: pass by value and use std::move [modernize-pass-by-value]
// CHECK-FIXES: #include <utility>
// CHECK-FIXES: A(ThreadId tid) : threadid(std::move(tid)) {}
