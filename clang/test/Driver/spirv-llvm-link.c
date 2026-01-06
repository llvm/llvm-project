// Check BC input
// RUN: mkdir -p %t
// RUN: touch %t/a.bc
// RUN: touch %t/b.bc
// RUN: %clang -### --target=spirv64 -emit-llvm %t/a.bc %t/b.bc 2>&1 | FileCheck --check-prefix=CHECK-TOOL-BC %s

// CHECK-TOOL-BC: "-cc1" {{.*}} "-o" "[[TMP1_BC:.+]]" "-x" "ir" "{{.*}}.bc"
// CHECK-TOOL-BC: "-cc1" {{.*}} "-o" "[[TMP2_BC:.+]]" "-x" "ir" "{{.*}}.bc"
// CHECK-TOOL-BC: llvm-link{{.*}} "-o" {{.*}} "[[TMP1_BC]]" "[[TMP2_BC]]"
// CHECK-TOOL-BC-NOT: llvm-link

// RUN: %clang -ccc-print-bindings --target=spirv64 -emit-llvm %t/a.bc %t/b.bc 2>&1 | FileCheck -check-prefix=CHECK-BINDINGS-BC %s

// CHECK-BINDINGS-BC: "spirv64" - "clang", inputs: ["{{.*}}.bc"], output: "[[TMP1_BINDINGS_BC:.+]]"
// CHECK-BINDINGS-BC: "spirv64" - "clang", inputs: ["{{.*}}.bc"], output: "[[TMP2_BINDINGS_BC:.+]]"
// CHECK-BINDINGS-BC: "spirv64" - "SPIR-V::Linker", inputs: ["[[TMP1_BINDINGS_BC]]", "[[TMP2_BINDINGS_BC]]"], output: "{{.*}}.bc"

// Check source input
// RUN: touch %t/foo.c
// RUN: touch %t/bar.c

// RUN: %clang -### --target=spirv64 -emit-llvm %t/foo.c %t/bar.c 2>&1 | FileCheck --check-prefix=CHECK-TOOL-SRC %s

// CHECK-TOOL-SRC: "-cc1" {{.*}} "-o" "[[TMP1_SRC_BC:.+]]" "-x" "c" "{{.*}}foo.c"
// CHECK-TOOL-SRC: "-cc1" {{.*}} "-o" "[[TMP2_SRC_BC:.+]]" "-x" "c" "{{.*}}bar.c"
// CHECK-TOOL-SRC: llvm-link{{.*}} "-o" {{.*}} "[[TMP1_SRC_BC]]" "[[TMP2_SRC_BC]]"
// CHECK-TOOL-SRC-NOT: llvm-link

// RUN: %clang -ccc-print-bindings --target=spirv64 -emit-llvm %t/foo.c %t/bar.c 2>&1 | FileCheck -check-prefix=CHECK-BINDINGS-SRC %s

// CHECK-BINDINGS-SRC: "spirv64" - "clang", inputs: ["{{.*}}foo.c"], output: "[[TMP1_BINDINGS_SRC_BC:.+]]"
// CHECK-BINDINGS-SRC: "spirv64" - "clang", inputs: ["{{.*}}bar.c"], output: "[[TMP2_BINDINGS_SRC_BC:.+]]"
// CHECK-BINDINGS-SRC: "spirv64" - "SPIR-V::Linker", inputs: ["[[TMP1_BINDINGS_SRC_BC]]", "[[TMP2_BINDINGS_SRC_BC]]"], output: "{{.*}}.bc"
