+// UNSUPPORTED: system-windows
+
+#include "print-unit.h"
+#include "syshead.h"
+
+void foo(int i);
+
+// RUN: rm -rf %t
+// RUN: mkdir -p %t
+// RUN: cd %t && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S/Inputs=INPUT_ROOT -fdebug-prefix-map=%S=SRC_ROOT -fdebug-prefix-map=%t=BUILD_ROOT -index-store-path %t/idx %S/print-unit-remapped.c -triple x86_64-apple-macosx10.8
+// RUN: c-index-test core -print-unit %t/idx | FileCheck %s
+
+// Relative paths should work as well - the unit name, main-path, and out-file should not change.
+// RUN: rm -rf %t
+// RUN: cd %S && %clang_cc1 -I %S/Inputs -isystem %S/Inputs/sys -fdebug-prefix-map=%S=SRC_ROOT -index-store-path %t/idx print-unit-remapped.c -o print-unit-remapped.c.o -triple x86_64-apple-macosx10.8
+// RUN: c-index-test core -print-unit %t/idx | FileCheck %s
+
+// CHECK: print-unit-remapped.c.o-20EK9G967JO97
