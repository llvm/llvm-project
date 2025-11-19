// RUN: %clang -### -ccc-print-phases --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PHASES
// PHASES: 0: input, "[[INPUT:.+]]", c
// PHASES: 1: preprocessor, {0}, cpp-output
// PHASES: 2: compiler, {1}, ir
// PHASES: 3: backend, {2}, assembler
// PHASES: 4: assembler, {3}, object
// PHASES: 5: linker, {4}, image

// RUN: %clang -### -ccc-print-bindings --target=spirv64-amd-amdhsa %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BINDINGS
// BINDINGS: # "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[OUTPUT:.+]]"
// BINDINGS: # "spirv64-amd-amdhsa" - "AMDGCN::Linker", inputs: ["[[OUTPUT]]"], output: "a.out"

// RUN: %clang -### --target=spirv64-amd-amdhsa %s -nogpulib -nogpuinc 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVOCATION
// INVOCATION: "-cc1" "-triple" "spirv64-amd-amdhsa" {{.*}}"-disable-llvm-optzns" {{.*}} "-o" "[[OUTPUT:.+]]" "-x" "c"
// INVOCATION: "{{.*}}llvm-link" "-o" "[[LINKED_OUTPUT:.+]]" "[[OUTPUT]]"
// INVOCATION: "{{.*}}llvm-spirv" "--spirv-max-version=1.6" "--spirv-ext=+all" "--spirv-allow-unknown-intrinsics" "--spirv-lower-const-expr" "--spirv-preserve-auxdata" "--spirv-debug-info-version=nonsemantic-shader-200" "[[LINKED_OUTPUT]]" "-o" "a.out"
