// Tests the clang-sycl-linker tool
//
// Test a simple case without arguments
// RUN: %clangxx -fsycl -emit-llvm -c %s -o %t.bc
// RUN: echo ' ' > %T/lib1.bc
// RUN: echo ' ' > %T/lib2.bc
// RUN: clang-sycl-linker --dry-run -triple spirv64 %t.bc --library-path=%T --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CMDS
// CMDS: "{{.*}}llvm-link{{.*}}" {{.*}}.bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CMDS-NEXT: "{{.*}}llvm-link{{.*}}" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}lib1.bc {{.*}}lib2.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CMDS-NEXT: "{{.*}}llvm-spirv{{.*}}" {{.*}}-o a.spv [[SECONDLLVMLINKOUT]].bc
