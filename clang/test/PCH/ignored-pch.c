// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -o %t.pch
// RUN: %clang %s -emit-ast -include-pch %t.pch -o %t.ll
// RUN: ls %t.pch | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Check that -ignore-pch causes -emit-pch and -include-pch options to be ignored.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH-ERROR %s
// RUN: ls %t.ll 2>&1 | FileCheck --check-prefix=CHECK-OBJ %s

// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -emit-ast %s -include-pch %t.pch -ignore-pch -o %t.ll
// RUN: not ls %t.ll 2>&1 | FileCheck --check-prefix=CHECK-OBJ-ERROR %s

// Check that -ignore-pch works for multiple PCH related options.
// Test with -building-pch-with-obj.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -building-pch-with-obj -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -Xclang -building-pch-with-obj -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fallow-pch-with-compiler-errors.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -fallow-pch-with-compiler-errors -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -Xclang -fallow-pch-with-compiler-errors -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fallow-pch-with-different-modules-cache-path.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -fallow-pch-with-different-modules-cache-path -o %t.pch
// RUN: %clang -S -emit-llvm %s -ignore-pch -include-pch %t.pch -Xclang -fallow-pch-with-different-modules-cache-path -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-codegen.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -fpch-codegen -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -fpch-codegen -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH-ERROR %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-debuginfo.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -fpch-debuginfo -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -fpch-debuginfo -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fpch-instantiate-templates.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -fpch-instantiate-templates -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fno-pch-timestamp.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -fno-pch-timestamp -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -Xclang -fno-pch-timestamp -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -fno-validate-pch.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -fno-validate-pch -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -Xclang -fno-validate-pch -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -relocatable-pch.
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -relocatable-pch -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -relocatable-pch -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s

// Test with -pch-through-hdrstop-create/-pch-through-hdrstop-use
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -ignore-pch -Xclang -pch-through-hdrstop-create -o %t.pch
// RUN: %clang -S -emit-llvm %s -include-pch %t.pch -ignore-pch -Xclang -pch-through-hdrstop-use -o %t.ll
// RUN: not ls %t.pch 2>&1 | FileCheck --check-prefix=CHECK-PCH %s
// RUN: ls %t.ll | FileCheck --check-prefix=CHECK-OBJ %s


// Test with AST dump output:
// RUN: rm -rf %t.pch %t.ll
// RUN: %clang -x c-header %S/Inputs/ignored-pch.h -o %t.pch
// RUN: %clang %s -include-pch %t.pch -Xclang -ast-dump-all -c | FileCheck --check-prefix=CHECK-AST-PCH %s
// RUN: %clang %s -include-pch %t.pch -ignore-pch -Xclang -ast-dump-all -c | FileCheck --check-prefix=CHECK-AST %s

// CHECK-PCH: ignored-pch.c.{{.*}}.pch
// CHECK-OBJ: ignored-pch.c.{{.*}}.ll
// CHECK-PCH-ERROR: ignored-pch.c.{{.*}}.pch{{'?}}: No such file or directory
// CHECK-OBJ-ERROR: ignored-pch.c.{{.*}}.ll{{'?}}: No such file or directory
// CHECK-AST-PCH: <undeserialized declarations>
// CHECK-AST-NOT: <undeserialized declarations>

#pragma hdrstop
#include "Inputs/ignored-pch.h"
int main() {
  return f();
}
