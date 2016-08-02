// RUN: %clang_cc1 -analyze -analyzer-display-progress %s 2>&1 | FileCheck %s

void f() {};
void g() {};
void h() {}

// CHECK: analyze_display_progress.c f
// CHECK: analyze_display_progress.c g
// CHECK: analyze_display_progress.c h