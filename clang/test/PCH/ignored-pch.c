// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Check that -ignore-pch causes -emit-pch and -include-pch options to be ignored.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -ignore-pch -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -ignore-pch -emit-llvm -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Check that -ignore-pch is passed through Driver.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -Xclang -emit-pch -o %t.pch
// RUN: %clang -S %s -include-pch %t.pch -Xclang -emit-llvm -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -Xclang -ignore-pch -Xclang -emit-pch -o %t.pch
// RUN: %clang -S %s -include-pch %t.pch -Xclang -ignore-pch -Xclang -emit-llvm -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Check that -ignore-pch works for multiple PCH related options.
// Test with -building-pch-with-obj.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -ignore-pch -emit-pch -building-pch-with-obj -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -ignore-pch -emit-llvm -building-pch-with-obj -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fallow-pch-with-compiler-errors.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -ignore-pch -emit-pch -fallow-pch-with-compiler-errors -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -ignore-pch -emit-llvm -fallow-pch-with-compiler-errors -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fallow-pch-with-different-modules-cache-path.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fallow-pch-with-different-modules-cache-path -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fallow-pch-with-different-modules-cache-path -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-codegen.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fpch-codegen -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fpch-codegen -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-debuginfo.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fpch-debuginfo -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fpch-debuginfo -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-instantiate-templates.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fpch-instantiate-templates -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fno-pch-timestamp.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fno-pch-timestamp -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fno-pch-timestamp -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fno-validate-pch.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -fno-validate-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -fno-validate-pch -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -relocatable-pch.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -relocatable-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -relocatable-pch -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -pch-through-hdrstop-create/-pch-through-hdrstop-use
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -pch-through-hdrstop-create -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -emit-llvm -pch-through-hdrstop-use -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with AST dump output:
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang_cc1 -x c-header %S/Inputs/ignored-pch.h -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -include-pch %t.pch -ast-dump-all | FileCheck --check-prefix=CHECK-AST-PCH %s
// RUN: %clang_cc1 %s -include-pch %t.pch -ignore-pch -ast-dump-all | FileCheck --check-prefix=CHECK-AST %s

// CHECK-PCH: ignored-pch.c.{{.*}}.pch
// CHECK-OBJ: ignored-pch.c.{{.*}}.ll
// CHECK-ERROR: ignored-pch.c.{{.*}}.pch{{'?}}: No such file or directory
// CHECK-AST-PCH: <undeserialized declarations>
// CHECK-AST-NOT: <undeserialized declarations>

#include "Inputs/ignored-pch.h"
#pragma hdrstop
int main() {
  return f();
}
