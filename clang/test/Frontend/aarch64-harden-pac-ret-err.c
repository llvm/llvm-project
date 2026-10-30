// RUN: not %clang_cc1 -fsyntax-only -triple aarch64 %s -mharden-pac-ret=foo 2>&1 | FileCheck %s

// CHECK: invalid value 'foo' in '-mharden-pac-ret=foo'
