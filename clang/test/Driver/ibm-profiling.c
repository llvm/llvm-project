// Check that -pg throws an error on z/OS.
// RUN: %clang -### 2>&1 --target=s390x-none-zos -S -pg %s | FileCheck -check-prefix=FAIL-PG-NAME %s
// FAIL-PG-NAME: error: unsupported option '-pg' for target 's390x-none-zos'

// Check that -p is still used when not linking on AIX.
// RUN: %clang -### 2>&1 --target=powerpc-ibm-aix7.1.0.0 -S -p -S %s \
// RUN:   | FileCheck --check-prefix=CHECK %s
// CHECK-NOT: warning: argument unused during compilation: '-p'

// Check precedence: -pg is unused when passed first on AIX.
// RUN: %clang -### 2>&1 --target=powerpc-ibm-aix7.1.0.0 --sysroot %S/Inputs/aix_ppc_tree -pg -p %s \
// RUN:        | FileCheck --check-prefix=CHECK2 %s
// CHECK2-NOT: warning: argument unused during compilation: '-p' [-Wunused-command-line-argument]
// CHECK2:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK2:     "[[SYSROOT]]/usr/lib{{/|\\\\}}mcrt0.o"
// CHECK2:     "-L[[SYSROOT]]/lib/profiled"
// CHECK2:     "-L[[SYSROOT]]/usr/lib/profiled"

// Check precedence: -p is unused when passed first on AIX.
// RUN: %clang -### 2>&1 --target=powerpc-ibm-aix7.1.0.0 --sysroot %S/Inputs/aix_ppc_tree -p -pg %s \
// RUN:        | FileCheck --check-prefix=CHECK3 %s
// CHECK3: warning: argument unused during compilation: '-p' [-Wunused-command-line-argument]
// CHECK3:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK3:     "[[SYSROOT]]/usr/lib{{/|\\\\}}gcrt0.o"
// CHECK3:     "-L[[SYSROOT]]/lib/profiled"
// CHECK3:     "-L[[SYSROOT]]/usr/lib/profiled"

