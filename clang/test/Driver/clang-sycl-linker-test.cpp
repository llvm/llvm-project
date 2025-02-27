// Tests the clang-sycl-linker tool.
//
// Test a simple case without arguments.
// RUN: %clangxx -emit-llvm -c %s -o %t_1.bc
// RUN: %clangxx -emit-llvm -c %s -o %t_2.bc
// RUN: clang-sycl-linker --dry-run -triple spirv64 %t_1.bc %t_2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE
// SIMPLE: "{{.*}}llvm-link{{.*}}" {{.*}}.bc {{.*}}.bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// SIMPLE-NEXT: "{{.*}}llvm-spirv{{.*}}" {{.*}}-o a.spv [[FIRSTLLVMLINKOUT]].bc
//
// Test that llvm-link is not called when only one input is present.
// RUN: clang-sycl-linker --dry-run -triple spirv64 %t_1.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SIMPLE-NO-LINK
// SIMPLE-NO-LINK: "{{.*}}llvm-spirv{{.*}}" {{.*}}-o a.spv {{.*}}.bc
//
// Test a simple case with device library files specified.
// RUN: touch %T/lib1.bc
// RUN: touch %T/lib2.bc
// RUN: clang-sycl-linker --dry-run -triple spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs=lib1.bc,lib2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBS
// DEVLIBS: "{{.*}}llvm-link{{.*}}" {{.*}}.bc {{.*}}.bc -o [[FIRSTLLVMLINKOUT:.*]].bc --suppress-warnings
// DEVLIBS-NEXT: "{{.*}}llvm-link{{.*}}" -only-needed [[FIRSTLLVMLINKOUT]].bc {{.*}}lib1.bc {{.*}}lib2.bc -o [[SECONDLLVMLINKOUT:.*]].bc --suppress-warnings
// DEVLIBS-NEXT: "{{.*}}llvm-spirv{{.*}}" {{.*}}-o a.spv [[SECONDLLVMLINKOUT]].bc
//
// Test a simple case with .o (fat object) as input.
// TODO: Remove this test once fat object support is added.
// RUN: %clangxx -c %s -o %t.o
// RUN: not clang-sycl-linker --dry-run -triple spirv64 %t.o -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
// FILETYPEERROR: Unsupported file type
//
// Test to see if device library related errors are emitted.
// RUN: not clang-sycl-linker --dry-run -triple spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs= -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR1
// DEVLIBSERR1: Number of device library files cannot be zero
// RUN: not clang-sycl-linker --dry-run -triple spirv64 %t_1.bc %t_2.bc --library-path=%T --device-libs=lib1.bc,lib2.bc,lib3.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEVLIBSERR2
// DEVLIBSERR2: '{{.*}}lib3.bc' SYCL device library file is not found
//
// Test if correct set of llvm-spirv options are emitted for windows environment.
// RUN: clang-sycl-linker --dry-run -triple spirv64 --is-windows-msvc-env %t_1.bc %t_2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LLVMOPTSWIN
// LLVMOPTSWIN: -spirv-debug-info-version=ocl-100 -spirv-allow-extra-diexpressions -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=
//
// Test if correct set of llvm-spirv options are emitted for linux environment.
// RUN: clang-sycl-linker --dry-run -triple spirv64  %t_1.bc %t_2.bc -o a.spv 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LLVMOPTSLIN
// LLVMOPTSLIN: -spirv-debug-info-version=nonsemantic-shader-200 -spirv-allow-unknown-intrinsics=llvm.genx. -spirv-ext=
