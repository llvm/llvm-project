// RUN: %clang_cc1 -triple i386-unknown-unknown %s -verify
// expected-no-diagnostics

using T1 [[gnu::regparm(2)]] = void(int);
