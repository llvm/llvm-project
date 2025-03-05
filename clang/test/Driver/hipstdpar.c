// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar --hipstdpar-path=/does/not/exist -nogpulib    \
// RUN:   -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-MISSING-LIB %s
// RUN: %clang -### --target=x86_64-unknown-linux-gnu \
// RUN:   --hipstdpar --hipstdpar-path=%S/Inputs/hipstdpar             \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   -nogpulib -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-COMPILE %s
// RUN: touch %t.o
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --hipstdpar %t.o 2>&1 | FileCheck --check-prefix=HIPSTDPAR-LINK %s

// HIPSTDPAR-MISSING-LIB: error: cannot find HIP Standard Parallelism Acceleration library; provide it via '--hipstdpar-path'
// HIPSTDPAR-COMPILE: "-x" "hip"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/thrust}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/rocprim}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/Inputs/hipstdpar}}"
// HIPSTDPAR-COMPILE: "-include" "hipstdpar_lib.hpp"
// HIPSTDPAR-LINK: "-rpath"
// HIPSTDPAR-LINK: "-l{{.*hip.*}}"
