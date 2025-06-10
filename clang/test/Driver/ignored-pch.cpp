// RUN: rm -rf %t
// RUN: mkdir -p %t

// Create PCH without -ignore-pch.
// RUN: %clang -x c++-header %S/../Modules/Inputs/codegen-flags/foo.h -### 2>&1 | FileCheck %s -check-prefix=CHECK-EMIT-PCH
// RUN: %clang -x c++-header %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo.h.pch
// RUN: %clang %s -include-pch %t/foo.h.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-INCLUDE-PCH
// RUN: %clang %s -emit-ast -include-pch %t/foo.h.pch -### 2>&1 | FileCheck %s -check-prefixes=CHECK-EMIT-PCH,CHECK-INCLUDE-PCH


// Create PCH with -ignore-pch.
// RUN: %clang -x c++-header -ignore-pch %S/../Modules/Inputs/codegen-flags/foo.h -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-PCH
// RUN: %clang %s -ignore-pch -include-pch  %t/foo.h.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-PCH
// RUN: %clang %s -ignore-pch -emit-ast -include-pch %t/foo.h.pch -### 2>&1 | FileCheck %s -check-prefix=CHECK-IGNORE-PCH

// CHECK-EMIT-PCH: -emit-pch
// CHECK-INCLUDE-PCH: -include-pch
// CHECK-IGNORE-PCH-NOT: -emit-pch
// CHECK-IGNORE-PCH-NOT: -include-pch
