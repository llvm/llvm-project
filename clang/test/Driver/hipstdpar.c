// XFAIL: target={{.*}}-apple{{.*}}
// XFAIL: target={{.*}}hexagon{{.*}}
// XFAIL: target={{.*}}-scei{{.*}}
// XFAIL: target={{.*}}-sie{{.*}}
// XFAIL: target={{.*}}-windows{{.*}}

// RUN: not %clang -### --hipstdpar -nogpulib -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-MISSING-LIB %s
// RUN: %clang -### --hipstdpar --hipstdpar-path=%S/Inputs/hipstdpar \
// RUN:   --hipstdpar-thrust-path=%S/Inputs/hipstdpar/thrust \
// RUN:   --hipstdpar-prim-path=%S/Inputs/hipstdpar/rocprim \
// RUN:   -nogpulib -nogpuinc --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=HIPSTDPAR-COMPILE %s
// RUN: touch %t.o
// RUN: %clang -### --hipstdpar %t.o 2>&1 | FileCheck --check-prefix=HIPSTDPAR-LINK %s

// HIPSTDPAR-MISSING-LIB: error: cannot find HIP Standard Parallelism Acceleration library; provide it via '--hipstdpar-path'
// HIPSTDPAR-COMPILE: "-x" "hip"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/thrust}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/rocprim}}"
// HIPSTDPAR-COMPILE: "-idirafter" "{{.*/Inputs/hipstdpar}}"
// HIPSTDPAR-COMPILE: "-include" "hipstdpar_lib.hpp"
// HIPSTDPAR-LINK: "-rpath"
// HIPSTDPAR-LINK: "-l{{.*hip.*}}"
