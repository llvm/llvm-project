// RUN: %clang_cc1 -isystem %S %s -fsyntax-only -verify 

#include <warn-in-system-header.h>
// expected-warning {{the cake is a lie}}
