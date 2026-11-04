// Check BC input with -S
// RUN: mkdir -p %t
// RUN: touch %t/a.bc
// RUN: touch %t/b.bc
// RUN: %clang -### --target=spirv64 -emit-llvm -S %t/a.bc %t/b.bc 2>&1 | FileCheck --check-prefix=CHECK-TOOL-BC %s

// CHECK-TOOL-BC: "-cc1" {{.*}} "-o" "{{.*}}.ll" "-x" "ir" "{{.*}}.bc"
// CHECK-TOOL-BC: "-cc1" {{.*}} "-o" "{{.*}}.ll" "-x" "ir" "{{.*}}.bc"
// CHECK-TOOL-BC-NOT: llvm-link

// RUN: %clang -ccc-print-bindings --target=spirv64 -emit-llvm -S %t/a.bc %t/b.bc 2>&1 | FileCheck -check-prefix=CHECK-BINDINGS-BC %s

// CHECK-BINDINGS-BC: "spirv64" - "clang", inputs: ["{{.*}}.bc"], output: "[[TMP1_BINDINGS_BC:.+]]"
// CHECK-BINDINGS-BC: "spirv64" - "clang", inputs: ["{{.*}}.bc"], output: "[[TMP2_BINDINGS_BC:.+]]"
// CHECK-BINDINGS-BC-NOT: SPIR-V::Linker

// Check source input with -S
// RUN: touch %t/foo.c
// RUN: touch %t/bar.c

// RUN: %clang -### --target=spirv64 -emit-llvm -S %t/foo.c %t/bar.c 2>&1 | FileCheck --check-prefix=CHECK-TOOL-SRC %s

// CHECK-TOOL-SRC: "-cc1" {{.*}} "-o" "{{.*}}.ll" "-x" "c" "{{.*}}foo.c"
// CHECK-TOOL-SRC: "-cc1" {{.*}} "-o" "{{.*}}.ll" "-x" "c" "{{.*}}bar.c"
// CHECK-TOOL-SRC-NOT: llvm-link

// RUN: %clang -ccc-print-bindings --target=spirv64 -emit-llvm -S %t/foo.c %t/bar.c 2>&1 | FileCheck -check-prefix=CHECK-BINDINGS-SRC %s

// CHECK-BINDINGS-SRC: "spirv64" - "clang", inputs: ["{{.*}}foo.c"], output: "{{.*}}.ll"
// CHECK-BINDINGS-SRC: "spirv64" - "clang", inputs: ["{{.*}}bar.c"], output: "{{.*}}.ll"
// CHECK-BINDINGS-SRC-NOT: SPIR-V::Linker
