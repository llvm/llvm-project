// Check SPIR-V support for LTO
// RUN: mkdir -p %t
// RUN: touch %t/a.cpp
// RUN: touch %t/b.cpp
// RUN: touch %t/a.o
// RUN: touch %t/b.o

// RUN: %clang -### --target=spirv64 -flto %t/a.cpp %t/b.cpp  -Xlinker --disable-verify 2>&1 | FileCheck --check-prefix=CHECK-POSITIVE-TOOL %s
// RUN: %clang -ccc-print-phases --target=spirv64 -flto %t/a.cpp %t/b.cpp 2>&1 | FileCheck --check-prefix=CHECK-POSITIVE-PHASES %s
// RUN: not %clang -### --target=spirv64 -flto %t/a.cpp %t/b.cpp --sycl-link 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

// RUN: %clang -### --target=spirv64 -flto %t/a.o %t/b.o  -Xlinker --disable-verify 2>&1 | FileCheck --check-prefix=CHECK-POSITIVE-TOOL-OBJ %s
// RUN: %clang -ccc-print-phases --target=spirv64 -flto %t/a.o %t/b.o 2>&1 | FileCheck --check-prefix=CHECK-POSITIVE-PHASES-OBJ %s
// RUN: not %clang -### --target=spirv64 -flto %t/a.o %t/b.o --sycl-link 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

// CHECK-POSITIVE-TOOL: llvm-lto{{.*}} "{{.*}}a-{{.*}}.o" "{{.*}}b-{{.*}}.o" "--disable-verify" "-o" "a.out" "-enable-lto-internalization=false"

// CHECK-POSITIVE-PHASES: 0: input, "{{.*}}a.cpp", c++
// CHECK-POSITIVE-PHASES: 1: preprocessor, {0}, c++-cpp-output
// CHECK-POSITIVE-PHASES: 2: compiler, {1}, ir
// CHECK-POSITIVE-PHASES: 3: backend, {2}, lto-bc
// CHECK-POSITIVE-PHASES: 4: input, "{{.*}}b.cpp", c++
// CHECK-POSITIVE-PHASES: 5: preprocessor, {4}, c++-cpp-output
// CHECK-POSITIVE-PHASES: 6: compiler, {5}, ir
// CHECK-POSITIVE-PHASES: 7: backend, {6}, lto-bc
// CHECK-POSITIVE-PHASES: 8: linker, {3, 7}, image

// CHECK-POSITIVE-TOOL-OBJ:  llvm-lto{{.*}} "{{.*}}a.o" "{{.*}}b.o" "--disable-verify" "-o" "a.out" "-enable-lto-internalization=false"

// CHECK-POSITIVE-PHASES-OBJ: 0: input, "{{.*}}a.o", object
// CHECK-POSITIVE-PHASES-OBJ: 1: input, "{{.*}}b.o", object
// CHECK-POSITIVE-PHASES-OBJ: 2: linker, {0, 1}, image

// CHECK-ERROR: 'spirv64': unable to pass LLVM bit-code files to linker

