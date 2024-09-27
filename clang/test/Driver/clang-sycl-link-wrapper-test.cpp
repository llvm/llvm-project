// Tests the clang-sycl-link-wrapper tool
//
// Test a simple case without arguments
// RUN: %clangxx -fsycl -emit-llvm -c %s -o %t.bc
// RUN: clang-sycl-link-wrapper --dry-run -triple spirv64 %t.bc --library-path=%S/Inputs -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CMDS
// CMDS: "{{.*}}llvm-link{{.*}}" {{.*}}.bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// CMDS-NEXT: "{{.*}}llvm-link{{.*}}" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}libsycl-crt.bc {{.*}}libsycl-complex.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// CMDS-NEXT: "{{.*}}llvm-spirv{{.*}}" {{.*}}-o a.spv [[SECONDLLVMLINKOUT]].bc
