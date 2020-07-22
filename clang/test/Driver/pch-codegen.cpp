// RUN: rm -rf %t
// RUN: mkdir -p %t

// Create PCH without codegen.
// RUN: %clang -x c++-header %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-cg.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-PCH-CREATE
// CHECK-PCH-CREATE: -emit-pch
// CHECK-PCH-CREATE-NOT: -fmodules-codegen
// CHECK-PCH-CREATE-NOT: -fmodules-debuginfo

// Create PCH with -fpch-codegen.
// RUN: %clang -x c++-header -fpch-codegen %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-cg.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-PCH-CODEGEN-CREATE
// CHECK-PCH-CODEGEN-CREATE: -emit-pch
// CHECK-PCH-CODEGEN-CREATE: -fmodules-codegen
// CHECK-PCH-CODEGEN-CREATE: "-x" "c++-header"
// CHECK-PCH-CODEGEN-CREATE-NOT: -fmodules-debuginfo

// Create PCH with -fpch-debuginfo.
// RUN: %clang -x c++-header -fpch-debuginfo %S/../Modules/Inputs/codegen-flags/foo.h -g -o %t/foo-di.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-PCH-DEBUGINFO-CREATE
// CHECK-PCH-DEBUGINFO-CREATE: -emit-pch
// CHECK-PCH-DEBUGINFO-CREATE: -fmodules-debuginfo
// CHECK-PCH-DEBUGINFO-CREATE: "-x" "c++-header"
// CHECK-PCH-DEBUGINFO-CREATE-NOT: -fmodules-codegen

// Create PCH's object file for -fpch-codegen.
// RUN: touch %t/foo-cg.pch
// RUN: %clang -c -fintegrated-as %t/foo-cg.pch -o %t/foo-cg.o -### 2>&1 | FileCheck %s -check-prefix=CHECK-PCH-CODEGEN-OBJ
// CHECK-PCH-CODEGEN-OBJ: -emit-obj
// CHECK-PCH-CODEGEN-OBJ: "-main-file-name" "foo-cg.pch"
// CHECK-PCH-CODEGEN-OBJ: "-o" "{{.*}}foo-cg.o"
// CHECK-PCH-CODEGEN-OBJ: "-x" "precompiled-header"

// Create PCH's object file for -fpch-debuginfo.
// RUN: touch %t/foo-di.pch
// RUN: %clang -c -fintegrated-as %t/foo-di.pch -g -o %t/foo-di.o -### 2>&1 | FileCheck %s -check-prefix=CHECK-PCH-DEBUGINFO-OBJ
// CHECK-PCH-DEBUGINFO-OBJ: -emit-obj
// CHECK-PCH-DEBUGINFO-OBJ: "-main-file-name" "foo-di.pch"
// CHECK-PCH-DEBUGINFO-OBJ: "-o" "{{.*}}foo-di.o"
// CHECK-PCH-DEBUGINFO-OBJ: "-x" "precompiled-header"
