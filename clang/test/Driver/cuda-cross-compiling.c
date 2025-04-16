// Tests the driver when targeting the NVPTX architecture directly without a
// host toolchain to perform CUDA mappings.

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
// ARGS-NEXT: ptxas{{.*}}"-m64" "-O0" "--gpu-name" "sm_61" "--output-file" "[[CUBIN:.+]].o" "[[PTX]].s" "-c"
// ARGS-NEXT: clang-nvlink-wrapper{{.*}}"-o" "a.out" "-arch" "sm_61"{{.*}}"[[CUBIN]].o"

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

// LINK: clang-nvlink-wrapper{{.*}}"-o" "a.out" "-arch" "sm_61"{{.*}}[[CUBIN:.+]].o

//
// Test to ensure that we enable handling global constructors in a freestanding
// Nvidia compilation.
//
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_70 %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=LOWERING %s
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_70 -flto -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=LOWERING-LTO %s

// LOWERING: -cc1" "-triple" "nvptx64-nvidia-cuda" {{.*}} "-mllvm" "--nvptx-lower-global-ctor-dtor"
// LOWERING: clang-nvlink-wrapper{{.*}} "-mllvm" "--nvptx-lower-global-ctor-dtor"
// LOWERING-LTO-NOT: "--nvptx-lower-global-ctor-dtor"

//
// Test passing arguments directly to nvlink.
//
// RUN: %clang -target nvptx64-nvidia-cuda -Wl,-v -Wl,a,b -march=sm_52 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINKER-ARGS %s

// LINKER-ARGS: clang-nvlink-wrapper{{.*}}"-v"{{.*}}"a" "b"

// Tests for handling a missing architecture.
//
// RUN: not %clang -target nvptx64-nvidia-cuda %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING %s
// RUN: not %clang -target nvptx64-nvidia-cuda -march=generic %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING %s

// MISSING: error: must pass in an explicit nvptx64 gpu architecture to 'ptxas'
// MISSING: error: must pass in an explicit nvptx64 gpu architecture to 'nvlink'

// Do not error when performing LTO.
//
// RUN: %clang -target nvptx64-nvidia-cuda -flto %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=MISSING-LTO %s

// MISSING-LTO-NOT: error: must pass in an explicit nvptx64 gpu architecture to 'nvlink'

// RUN: %clang -target nvptx64-nvidia-cuda -flto -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_52 -march=generic -flto -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=GENERIC %s

// GENERIC-NOT: -cc1" "-triple" "nvptx64-nvidia-cuda" {{.*}} "-target-cpu"

//
// Test forwarding the necessary +ptx feature.
//
// RUN: %clang -target nvptx64-nvidia-cuda --cuda-feature=+ptx63 -march=sm_52 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FEATURE %s

// FEATURE: clang-nvlink-wrapper{{.*}}"--plugin-opt=-mattr=+ptx63"

//
// Test including the libc startup files and libc
//
// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_61 -stdlib -startfiles \
// RUN:   -nogpulib -nogpuinc -### %s 2>&1 | FileCheck -check-prefix=STARTUP %s

// STARTUP: clang-nvlink-wrapper{{.*}}"-lc" "-lm" "{{.*}}crt1.o"
