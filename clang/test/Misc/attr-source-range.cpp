// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s

void f(int i) __attribute__((format_arg(1)));
// CHECK: attr-source-range.cpp:3:30:{3:41-3:42}{3:8-3:13}

void g(int i, ...) __attribute__((format(printf, 1, 1)));
// CHECK: attr-source-range.cpp:6:35:{6:50-6:51}{6:8-6:13}

int h(void) __attribute__((returns_nonnull));
// CHECK: attr-source-range.cpp:9:28:{9:1-9:4}

void i(int j) __attribute__((nonnull(1)));
// CHECK: attr-source-range.cpp:12:30:{12:38-12:39}{12:8-12:13}

void j(__attribute__((nonnull)) int i);
// CHECK: attr-source-range.cpp:15:23:{15:8-15:38}
