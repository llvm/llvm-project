// RUN: %clang -### -ccc-print-phases --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PHASES
// PHASES: 0: input, "[[INPUT:.+]]", c
// PHASES: 1: preprocessor, {0}, cpp-output
// PHASES: 2: compiler, {1}, ir
// PHASES: 3: backend, {2}, ir
// PHASES: 4: linker, {3}, image

// RUN: %clang -### -ccc-print-phases -use-spirv-backend --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PHASES

// RUN: %clang -### -ccc-print-bindings --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BINDINGS
// BINDINGS: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[OUTPUT:.+]]"
// BINDINGS: # "spirv64-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN: %clang -### -ccc-print-bindings -use-spirv-backend --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BINDINGS

// RUN: %clang -### --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVOCATION
// INVOCATION: "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}}"-disable-llvm-optzns" {{.*}} "-o" "[[OUTPUT:.+]]" "-x" "c"
// INVOCATION: "{{.*}}llvm-link" "-o" "[[LINKED_OUTPUT:.+]]" "[[OUTPUT]]"
// INVOCATION: "{{.*}}llvm-spirv" "--spirv-max-version=1.6" "--spirv-ext=+all" "--spirv-allow-unknown-intrinsics" "--spirv-lower-const-expr" "--spirv-preserve-auxdata" "--spirv-debug-info-version=nonsemantic-shader-200" "[[LINKED_OUTPUT]]" "-o" "a.out"

// RUN: %clang -### -use-spirv-backend --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVOCATION-SPIRV-BACKEND
// INVOCATION-SPIRV-BACKEND: "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}}"-disable-llvm-optzns" {{.*}} "-o" "[[OUTPUT:.+]]" "-x" "c"
// INVOCATION-SPIRV-BACKEND: "{{.*}}llvm-link" "-o" "[[LINKED_OUTPUT:.+]]" "[[OUTPUT]]"
// INVOCATION-SPIRV-BACKEND: "-cc1" "-triple=spirv64-amd-amdhsa" "-emit-obj" {{.*}} "[[LINKED_OUTPUT]]" "-o" "a.out"
