// Needs chmod
// UNSUPPORTED: system-windows
//
// RUN: chmod -R 755 %t; rm -rf %t && mkdir -p %t/noaccess %t/haveaccess
// RUN: echo "int test();" > %t/haveaccess/test.h
// RUN: chmod 000 %t/noaccess
// RUN: %clang_cc1 -fsyntax-only -I %t/noaccess -I %t/haveaccess -verify %s

#include "test.h" // expected-warning {{cannot access file}}
