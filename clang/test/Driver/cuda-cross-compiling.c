// Tests the driver when targeting the NVPTX architecture directly without a
// host toolchain to perform CUDA mappings.

// REQUIRES: nvptx-registered-target

//
// Test the generated phases when targeting NVPTX.
//
// RUN: %clang -target nvptx64-nvidia-cuda -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=PHASES %s

//      PHASES: 0: input, "[[INPUT:.+]]", c
// PHASES-NEXT: 1: preprocessor, {0}, cpp-output
// PHASES-NEXT: 2: compiler, {1}, ir
// PHASES-NEXT: 3: backend, {2}, assembler
// PHASES-NEXT: 4: assembler, {3}, object
// PHASES-NEXT: 5: linker, {4}, image

//
// Test the generated bindings when targeting NVPTX.
//
// RUN: %clang -target nvptx64-nvidia-cuda -ccc-print-bindings %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BINDINGS %s

//      BINDINGS: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX:.+]].s"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX]].s"], output: "[[CUBIN:.+]].o"
// BINDINGS-NEXT: "nvptx64-nvidia-cuda" - "NVPTX::Linker", inputs: ["[[CUBIN]].o"], output: "a.out"

//
// Test the generated arguments to the CUDA binary utils when targeting NVPTX. 
// Ensure that the '.o' files are converted to '.cubin' if produced internally.
//
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_61 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARGS %s

//      ARGS: -cc1" "-triple" "nvptx64-nvidia-cuda" "-S" {{.*}} "-target-cpu" "sm_61" "-target-feature" "+ptx{{[0-9]+}}" {{.*}} "-o" "[[PTX:.+]].s"
// ARGS-NEXT: ptxas{{.*}}"-m64" "-O0" "--gpu-name" "sm_61" "--output-file" "[[CUBIN:.+]].cubin" "[[PTX]].s" "-c"
// ARGS-NEXT: nvlink{{.*}}"-o" "a.out" "-arch" "sm_61" {{.*}} "[[CUBIN]].cubin"

//
// Test the generated arguments to the CUDA binary utils when targeting NVPTX. 
// Ensure that we emit '.o' files if compiled with '-c'
//
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_61 -c -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OBJECT %s
// RUN: %clang -target nvptx64-nvidia-cuda -save-temps -march=sm_61 -c -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OBJECT %s

//      OBJECT: -cc1" "-triple" "nvptx64-nvidia-cuda" "-S" {{.*}} "-target-cpu" "sm_61" "-target-feature" "+ptx{{[0-9]+}}" {{.*}} "-o" "[[PTX:.+]].s"
// OBJECT-NEXT: ptxas{{.*}}"-m64" "-O0" "--gpu-name" "sm_61" "--output-file" "[[OBJ:.+]].o" "[[PTX]].s" "-c"

//
// Test the generated arguments to the CUDA binary utils when targeting NVPTX. 
// Ensure that we copy input '.o' files to '.cubin' files when linking.
//
// RUN: touch %t.o
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_61 -### %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s

// LINK: nvlink{{.*}}"-o" "a.out" "-arch" "sm_61" {{.*}} "{{.*}}.cubin"

//
// Test the generated arguments default to a value with no architecture. 
//
// RUN: %clang -target nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s

//      DEFAULT: -cc1" "-triple" "nvptx64-nvidia-cuda" "-S" {{.*}} "-target-cpu" "sm_35" "-target-feature" "+ptx{{[0-9]+}}" {{.*}} "-o" "[[PTX:.+]].s"
// DEFAULT-NEXT: ptxas{{.*}}"-m64" "-O0" "--gpu-name" "sm_35" "--output-file" "[[CUBIN:.+]].cubin" "[[PTX]].s" "-c"
// DEFAULT-NEXT: nvlink{{.*}}"-o" "a.out" "-arch" "sm_35" {{.*}} "[[CUBIN]].cubin"
