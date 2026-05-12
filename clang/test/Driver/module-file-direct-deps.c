// Test -fmodule-file-deps= with explicit values.
// RUN: %clang -c %s -o %t.o -MMD -MT dependencies -MF %t.d -fmodule-file-deps=direct -### 2> %t
// RUN: FileCheck %s -input-file=%t
// CHECK: -dependency-file
// CHECK: "-module-file-deps=direct"

// RUN: %clang -c %s -o %t.o -MMD -MT dependencies -MF %t.d -fmodule-file-deps=all -### 2> %t
// RUN: FileCheck %s -check-prefix=CHECK-ALL -input-file=%t
// CHECK-ALL: -dependency-file
// CHECK-ALL: "-module-file-deps=all"

// RUN: %clang -c %s -o %t.o -MMD -MT dependencies -MF %t.d -fmodule-file-deps=none -### 2> %t
// RUN: FileCheck %s -check-prefix=CHECK-NONE -input-file=%t
// CHECK-NONE: -dependency-file
// CHECK-NONE-NOT: -module-file-deps=

// Test legacy -fmodule-file-deps maps to =all.
// RUN: %clang -c %s -o %t.o -MMD -MT dependencies -MF %t.d -fmodule-file-deps -### 2> %t
// RUN: FileCheck %s -check-prefix=CHECK-LEGACY -input-file=%t
// CHECK-LEGACY: -dependency-file
// CHECK-LEGACY: "-module-file-deps=all"

// Test legacy -fno-module-file-deps suppresses module deps.
// RUN: %clang -c %s -o %t.o -MMD -MT dependencies -MF %t.d \
// RUN:     -fmodule-file-deps -fno-module-file-deps -### 2> %t
// RUN: FileCheck %s -check-prefix=CHECK-LEGACY-NEG -input-file=%t
// CHECK-LEGACY-NEG: -dependency-file
// CHECK-LEGACY-NEG-NOT: -module-file-deps=
