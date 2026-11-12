// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -E -o %t/diamond.mi -frewrite-imports
// RUN: FileCheck %s --input-file %t/diamond.mi
// RUN: %clang_cc1 -fmodules %t/diamond.mi -I. -verify

// CHECK: {{^}}#pragma clang module build diamond_top
// CHECK: {{^}}module diamond_top {
// CHECK: {{^}}#pragma clang module contents

// FIXME: @import does not work under -frewrite-includes / -frewrite-imports
// because we disable it when macro expansion is disabled.
#include "diamond_bottom.h"

// expected-no-diagnostics
void test_diamond(int i, float f, double d, char c) {
  top(&i);
  left(&f);
  right(&d);
  bottom(&c);
  top_left(&c);
  left_and_right(&i);
  struct left_and_right lr;
  lr.left = 17;
}

