// General tests that ld invocations on AIX targets are sane. Note that we use
// sysroot to make these tests independent of the host system.

// Check powerpc-ibm-aix7.1.0.0, 32-bit.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32 %s
// CHECK-LD32-NOT: warning:
// CHECK-LD32:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOT: "-bnso"
// CHECK-LD32:     "-b32"
// CHECK-LD32:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOT: "-lc++"
// CHECK-LD32-NOT: "-lc++abi"
// CHECK-LD32:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOT: "--as-needed"
// CHECK-LD32:     "-lunwind"
// CHECK-LD32-NOT: "--no-as-needed"
// CHECK-LD32-NOT: "-lm"
// CHECK-LD32:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64 %s
// CHECK-LD64-NOT: warning:
// CHECK-LD64:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOT: "-bnso"
// CHECK-LD64:     "-b64"
// CHECK-LD64:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOT: "-lc++"
// CHECK-LD64-NOT: "-lc++abi"
// CHECK-LD64:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NOT: "--as-needed"
// CHECK-LD64:     "-lunwind"
// CHECK-LD64-NOT: "--no-as-needed"
// CHECK-LD64-NOT: "-lm"
// CHECK-LD64:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Enable POSIX thread support.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -pthread \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-PTHREAD %s
// CHECK-LD32-PTHREAD-NOT: warning:
// CHECK-LD32-PTHREAD:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-PTHREAD:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-PTHREAD:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-PTHREAD:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-PTHREAD-NOT: "-bnso"
// CHECK-LD32-PTHREAD:     "-b32"
// CHECK-LD32-PTHREAD:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-PTHREAD-NOT: "-lc++"
// CHECK-LD32-PTHREAD-NOT: "-lc++abi"
// CHECK-LD32-PTHREAD:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-PTHREAD-NOT: "--as-needed"
// CHECK-LD32-PTHREAD:     "-lunwind"
// CHECK-LD32-PTHREAD-NOT: "--no-as-needed"
// CHECK-LD32-PTHREAD:     "-lpthreads"
// CHECK-LD32-PTHREAD-NOT: "-lm"
// CHECK-LD32-PTHREAD:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. POSIX thread alias.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -pthreads \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-PTHREAD %s
// CHECK-LD64-PTHREAD-NOT: warning:
// CHECK-LD64-PTHREAD:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-PTHREAD:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-PTHREAD:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-PTHREAD:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-PTHREAD-NOT: "-bnso"
// CHECK-LD64-PTHREAD:     "-b64"
// CHECK-LD64-PTHREAD:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-PTHREAD-NOT: "-lc++"
// CHECK-LD64-PTHREAD-NOT: "-lc++abi"
// CHECK-LD64-PTHREAD:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-PTHREAD-NOT: "--as-needed"
// CHECK-LD64-PTHREAD:     "-lunwind"
// CHECK-LD64-PTHREAD-NOT: "--no-as-needed"
// CHECK-LD64-PTHREAD:     "-lpthreads"
// CHECK-LD64-PTHREAD-NOT: "-lm"
// CHECK-LD64-PTHREAD:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Disable POSIX thread support.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -pthread -no-pthread\
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOPTHREAD %s
// CHECK-LD32-NOPTHREAD-NOT: warning:
// CHECK-LD32-NOPTHREAD:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOPTHREAD:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NOPTHREAD:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOPTHREAD:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOPTHREAD-NOT: "-bnso"
// CHECK-LD32-NOPTHREAD:     "-b32"
// CHECK-LD32-NOPTHREAD:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOPTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOPTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOPTHREAD-NOT: "-lc++"
// CHECK-LD32-NOPTHREAD-NOT: "-lc++abi"
// CHECK-LD32-NOPTHREAD:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOPTHREAD-NOT: "--as-needed"
// CHECK-LD32-NOPTHREAD:     "-lunwind"
// CHECK-LD32-NOPTHREAD-NOT: "--no-as-needed"
// CHECK-LD32-NOPTHREAD-NOT: "-lm"
// CHECK-LD32-NOPTHREAD:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Enable profiling.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -p \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-PROF %s
// CHECK-LD32-PROF-NOT: warning:
// CHECK-LD32-PROF:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-PROF:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-PROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-PROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-PROF-NOT: "-bnso"
// CHECK-LD32-PROF:     "-b32"
// CHECK-LD32-PROF:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-PROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}mcrt0.o"
// CHECK-LD32-PROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-PROF-NOT: "-lc++"
// CHECK-LD32-PROF-NOT: "-lc++abi"
// CHECK-LD32-PROF:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-PROF-NOT: "--as-needed"
// CHECK-LD32-PROF:     "-lunwind"
// CHECK-LD32-PROF-NOT: "--no-as-needed"
// CHECK-LD32-PROF-NOT: "-lm"
// CHECK-LD32-PROF:     "-lc"
// CHECK-LD32-PROF:     "-L[[SYSROOT]]/lib/profiled"
// CHECK-LD32-PROF:     "-L[[SYSROOT]]/usr/lib/profiled"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. Enable profiling.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -p \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-PROF %s
// CHECK-LD64-PROF-NOT: warning:
// CHECK-LD64-PROF:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-PROF:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-PROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-PROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-PROF-NOT: "-bnso"
// CHECK-LD64-PROF:     "-b64"
// CHECK-LD64-PROF:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-PROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}mcrt0_64.o"
// CHECK-LD64-PROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-PROF-NOT: "-lc++"
// CHECK-LD64-PROF-NOT: "-lc++abi"
// CHECK-LD64-PROF:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-PROF-NOT: "--as-needed"
// CHECK-LD64-PROF:     "-lunwind"
// CHECK-LD64-PROF-NOT: "--no-as-needed"
// CHECK-LD64-PROF-NOT: "-lm"
// CHECK-LD64-PROF:     "-lc"
// CHECK-LD64-PROF:     "-L[[SYSROOT]]/lib/profiled"
// CHECK-LD64-PROF:     "-L[[SYSROOT]]/usr/lib/profiled

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Enable g-profiling.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -pg \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-GPROF %s
// CHECK-LD32-GPROF-NOT: warning:
// CHECK-LD32-GPROF:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-GPROF:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-GPROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-GPROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-GPROF-NOT: "-bnso"
// CHECK-LD32-GPROF:     "-b32"
// CHECK-LD32-GPROF:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-GPROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}gcrt0.o"
// CHECK-LD32-GPROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-GPROF-NOT: "-lc++"
// CHECK-LD32-GPROF-NOT: "-lc++abi"
// CHECK-LD32-GPROF:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-GPROF-NOT: "--as-needed"
// CHECK-LD32-GPROF:     "-lunwind"
// CHECK-LD32-GPROF-NOT: "--no-as-needed"
// CHECK-LD32-GPROF-NOT: "-lm"
// CHECK-LD32-GPROF:     "-lc"
// CHECK-LD32-GPROF:     "-L[[SYSROOT]]/lib/profiled"
// CHECK-LD32-GPROF:     "-L[[SYSROOT]]/usr/lib/profiled"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. Enable g-profiling.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -pg \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-GPROF %s
// CHECK-LD64-GPROF-NOT: warning:
// CHECK-LD64-GPROF:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-GPROF:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-GPROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-GPROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-GPROF-NOT: "-bnso"
// CHECK-LD64-GPROF:     "-b64"
// CHECK-LD64-GPROF:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-GPROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}gcrt0_64.o"
// CHECK-LD64-GPROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-GPROF-NOT: "-lc++"
// CHECK-LD64-GPROF-NOT: "-lc++abi"
// CHECK-LD64-GPROF:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-GPROF-NOT: "--as-needed"
// CHECK-LD64-GPROF:     "-lunwind"
// CHECK-LD64-GPROF-NOT: "--no-as-needed"
// CHECK-LD64-GPROF-NOT: "-lm"
// CHECK-LD64-GPROF:     "-lc"
// CHECK-LD64-GPROF:     "-L[[SYSROOT]]/lib/profiled"
// CHECK-LD64-GPROF:     "-L[[SYSROOT]]/usr/lib/profiled"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Static linking.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -static \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-STATIC %s
// CHECK-LD32-STATIC-NOT: warning:
// CHECK-LD32-STATIC:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-STATIC:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-STATIC:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-STATIC:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-STATIC:     "-bnso"
// CHECK-LD32-STATIC:     "-b32"
// CHECK-LD32-STATIC:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-STATIC:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-STATIC:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-STATIC-NOT: "-lc++"
// CHECK-LD32-STATIC-NOT: "-lc++abi"
// CHECK-LD32-STATIC:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-STATIC-NOT: "--as-needed"
// CHECK-LD32-STATIC-NOT: "-lunwind"
// CHECK-LD32-STATIC-NOT: "--no-as-needed"
// CHECK-LD32-STATIC-NOT: "-lm"
// CHECK-LD32-STATIC:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Library search path.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -L%S/Inputs/aix_ppc_tree/powerpc-ibm-aix7.1.0.0 \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-LIBP %s
// CHECK-LD32-LIBP-NOT: warning:
// CHECK-LD32-LIBP:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-LIBP:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-LIBP:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-LIBP:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-LIBP-NOT: "-bnso"
// CHECK-LD32-LIBP:     "-b32"
// CHECK-LD32-LIBP:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-LIBP:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-LIBP:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-LIBP:     "-L[[SYSROOT]]/powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-LIBP-NOT: "-lc++"
// CHECK-LD32-LIBP-NOT: "-lc++abi"
// CHECK-LD32-LIBP:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-LIBP-NOT: "--as-needed"
// CHECK-LD32-LIBP:     "-lunwind"
// CHECK-LD32-LIBP-NOT: "--no-as-needed"
// CHECK-LD32-LIBP-NOT: "-lm"
// CHECK-LD32-LIBP:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. nostdlib.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostdlib \
// RUN:        -pthread \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NO-STD-LIB %s
// CHECK-LD32-NO-STD-LIB-NOT: warning:
// CHECK-LD32-NO-STD-LIB:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NO-STD-LIB:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NO-STD-LIB:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NO-STD-LIB:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NO-STD-LIB-NOT: "-bnso"
// CHECK-LD32-NO-STD-LIB:     "-b32"
// CHECK-LD32-NO-STD-LIB:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NO-STD-LIB-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NO-STD-LIB-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NO-STD-LIB-NOT: "-lc++"
// CHECK-LD32-NO-STD-LIB-NOT: "-lc++abi"
// CHECK-LD32-NO-STD-LIB-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NO-STD-LIB-NOT: "--as-needed"
// CHECK-LD32-NO-STD-LIB-NOT: "-lunwind"
// CHECK-LD32-NO-STD-LIB-NOT: "--no-as-needed"
// CHECK-LD32-NO-STD-LIB-NOT: "-lpthreads"
// CHECK-LD32-NO-STD-LIB-NOT: "-lm"
// CHECK-LD32-NO-STD-LIB-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. nodefaultlibs.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nodefaultlibs \
// RUN:        -pthread \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NO-DEFAULT-LIBS %s
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: warning:
// CHECK-LD64-NO-DEFAULT-LIBS:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NO-DEFAULT-LIBS:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-bnso"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-b64"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NO-DEFAULT-LIBS:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NO-DEFAULT-LIBS:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lc++"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lc++abi"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "--as-needed"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lunwind"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "--no-as-needed"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lpthreads"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lm"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. 'bcdtors' and argument order.
// RUN: %clang %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -Wl,-bnocdtors \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-ARG-ORDER %s
// CHECK-LD32-ARG-ORDER:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-ARG-ORDER:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-ARG-ORDER:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-ARG-ORDER:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-ARG-ORDER-NOT: "-bnso"
// CHECK-LD32-ARG-ORDER:     "-b32"
// CHECK-LD32-ARG-ORDER:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-ARG-ORDER:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-ARG-ORDER:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-ARG-ORDER:     "-bcdtors:all:0:s"
// CHECK-LD32-ARG-ORDER:     "-bnocdtors"
// CHECK-LD32-ARG-ORDER-NOT: "-bcdtors:all:0:s"
// CHECK-LD32-ARG-ORDER-NOT: "-lc++"
// CHECK-LD32-ARG-ORDER-NOT: "-lc++abi"
// CHECK-LD32-ARG-ORDER:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-ARG-ORDER-NOT: "--as-needed"
// CHECK-LD32-ARG-ORDER:     "-lunwind"
// CHECK-LD32-ARG-ORDER-NOT: "--no-as-needed"
// CHECK-LD32-ARG-ORDER-NOT: "-lm"
// CHECK-LD32-ARG-ORDER:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. 'bcdtors' and argument order.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -Wl,-bnocdtors \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-CXX-ARG-ORDER %s
// CHECK-LD32-CXX-ARG-ORDER:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-CXX-ARG-ORDER:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-CXX-ARG-ORDER:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-CXX-ARG-ORDER:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-CXX-ARG-ORDER-NOT: "-bnso"
// CHECK-LD32-CXX-ARG-ORDER:     "-b32"
// CHECK-LD32-CXX-ARG-ORDER:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-CXX-ARG-ORDER:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-CXX-ARG-ORDER:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-CXX-ARG-ORDER:     "-bcdtors:all:0:s"
// CHECK-LD32-CXX-ARG-ORDER:     "-bnocdtors"
// CHECK-LD32-CXX-ARG-ORDER-NOT: "-bcdtors:all:0:s"
// CHECK-LD32-CXX-ARG-ORDER:     "-lc++"
// CHECK-LD32-CXX-ARG-ORDER:     "-lc++abi"
// CHECK-LD32-CXX-ARG-ORDER:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-CXX-ARG-ORDER-NOT: "--as-needed"
// CHECK-LD32-CXX-ARG-ORDER:     "-lunwind"
// CHECK-LD32-CXX-ARG-ORDER-NOT: "--no-as-needed"
// CHECK-LD32-CXX-ARG-ORDER:     "-lm"
// CHECK-LD32-CXX-ARG-ORDER:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. lc++ and lc order.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-CXX-ARG-LCXX %s
// CHECK-LD32-CXX-ARG-LCXX:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-CXX-ARG-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-CXX-ARG-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-CXX-ARG-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-CXX-ARG-LCXX:     "-b32"
// CHECK-LD32-CXX-ARG-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-CXX-ARG-LCXX:     "-lc++"
// CHECK-LD32-CXX-ARG-LCXX:     "-lc++abi"
// CHECK-LD32-CXX-ARG-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-CXX-ARG-LCXX-NOT: "--as-needed"
// CHECK-LD32-CXX-ARG-LCXX:     "-lunwind"
// CHECK-LD32-CXX-ARG-LCXX-NOT: "--no-as-needed"
// CHECK-LD32-CXX-ARG-LCXX:     "-lm"
// CHECK-LD32-CXX-ARG-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. lc++ and lc order.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-CXX-ARG-LCXX %s
// CHECK-LD64-CXX-ARG-LCXX:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-CXX-ARG-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-CXX-ARG-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-CXX-ARG-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-CXX-ARG-LCXX:     "-b64"
// CHECK-LD64-CXX-ARG-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-CXX-ARG-LCXX:     "-lc++"
// CHECK-LD64-CXX-ARG-LCXX:     "-lc++abi"
// CHECK-LD64-CXX-ARG-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-CXX-ARG-LCXX-NOT: "--as-needed"
// CHECK-LD64-CXX-ARG-LCXX:     "-lunwind"
// CHECK-LD64-CXX-ARG-LCXX-NOT: "--no-as-needed"
// CHECK-LD64-CXX-ARG-LCXX:     "-lm"
// CHECK-LD64-CXX-ARG-LCXX:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nodefaultlibs.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nodefaultlibs \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NODEFLIB-LCXX %s
// CHECK-LD32-NODEFLIB-LCXX:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NODEFLIB-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NODEFLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NODEFLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NODEFLIB-LCXX:     "-b32"
// CHECK-LD32-NODEFLIB-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lc++"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lc++abi"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "--as-needed"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lunwind"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "--no-as-needed"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lm"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nodefaultlibs.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nodefaultlibs \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NODEFLIB-LCXX %s
// CHECK-LD64-NODEFLIB-LCXX:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NODEFLIB-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-NODEFLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NODEFLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NODEFLIB-LCXX:     "-b64"
// CHECK-LD64-NODEFLIB-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lc++"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lc++abi"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "--as-needed"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lunwind"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "--no-as-needed"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lm"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nostdlib.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostdlib \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTDLIB-LCXX %s
// CHECK-LD32-NOSTDLIB-LCXX:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTDLIB-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NOSTDLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTDLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTDLIB-LCXX:     "-b32"
// CHECK-LD32-NOSTDLIB-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lc++"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lc++abi"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "--as-needed"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lunwind"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "--no-as-needed"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lm"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostdlib.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostdlib \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NOSTDLIB-LCXX %s
// CHECK-LD64-NOSTDLIB-LCXX:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTDLIB-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-NOSTDLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTDLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTDLIB-LCXX:     "-b64"
// CHECK-LD64-NOSTDLIB-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lc++"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lc++abi"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "--as-needed"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lunwind"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "--no-as-needed"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lm"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nostdlib++.
// Note: crti is still linked for initialization/finalization functionality.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostdlib++ \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTDLIBXX-LCXX %s
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-b32"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTDLIBXX-LCXX-NOT: "-lc++"
// CHECK-LD32-NOSTDLIBXX-LCXX-NOT: "-lc++abi"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOSTDLIBXX-LCXX-NOT: "--as-needed"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-lunwind"
// CHECK-LD32-NOSTDLIBXX-LCXX-NOT: "--no-as-needed"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-lm"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostdlib++.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostdlib++ \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NOSTDLIBXX-LCXX %s
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-b64"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTDLIBXX-LCXX-NOT: "-lc++"
// CHECK-LD64-NOSTDLIBXX-LCXX-NOT: "-lc++abi"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NOSTDLIBXX-LCXX-NOT: "--as-needed"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-lunwind"
// CHECK-LD64-NOSTDLIBXX-LCXX-NOT: "--no-as-needed"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-lm"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 32-bit. -nostartfiles.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostartfiles \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTARTFILES-LCXX %s
// CHECK-LD32-NOSTARTFILES-LCXX:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTARTFILES-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-b32"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lc++"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lc++abi"
// CHECK-LD32-NOSTARTFILES-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "--as-needed"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lunwind"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "--no-as-needed"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lm"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostartfiles.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -nostartfiles \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTARTFILES-LCXX %s
// CHECK-LD64-NOSTARTFILES-LCXX:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTARTFILES-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-b64"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lc++"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lc++abi"
// CHECK-LD64-NOSTARTFILES-LCXX:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "--as-needed"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lunwind"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "--no-as-needed"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lm"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        -stdlib=libstdc++ -nostdinc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-LIBSTDCXX %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx %s 2>&1 -### \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        -stdlib=libstdc++ -nostdinc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-LIBSTDCXX %s
// CHECK-LD-LIBSTDCXX: LLVM ERROR: linking libstdc++ unimplemented on AIX

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -shared.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED %s

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -shared (with exp option strings in other opt).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -Wl,-Z/expall/expfull/a-bE:/a-bexport:/ \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED %s

// CHECK-LD32-SHARED:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-SHARED:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-SHARED:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-SHARED:     "{{.*}}llvm-nm"
// CHECK-LD32-SHARED:     "--export-symbols"
// CHECK-LD32-SHARED:     "-X" "32"
// CHECK-LD32-SHARED:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-SHARED:     "-bM:SRE"
// CHECK-LD32-SHARED:     "-bnoentry"
// CHECK-LD32-SHARED:     "-b32"
// CHECK-LD32-SHARED:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-SHARED:     "-lc++"
// CHECK-LD32-SHARED:     "-lc++abi"
// CHECK-LD32-SHARED:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-SHARED-NOT: "--as-needed"
// CHECK-LD32-SHARED:     "-lunwind"
// CHECK-LD32-SHARED-NOT: "--no-as-needed"
// CHECK-LD32-SHARED:     "-lm"
// CHECK-LD32-SHARED:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -shared with export list.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Wl,-bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED-EXPORTS %s

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -shared with export list (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED-EXPORTS %s

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -shared with export list (-Xlinker variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Xlinker -bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED-EXPORTS %s

// CHECK-LD32-SHARED-EXPORTS:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-SHARED-EXPORTS:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-SHARED-EXPORTS:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-SHARED-EXPORTS-NOT: "{{.*}}llvm-nm"
// CHECK-LD32-SHARED-EXPORTS-NOT: "-X"
// CHECK-LD32-SHARED-EXPORTS-NOT: "32"
// CHECK-LD32-SHARED-EXPORTS:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-SHARED-EXPORTS:     "-bM:SRE"
// CHECK-LD32-SHARED-EXPORTS:     "-bnoentry"
// CHECK-LD32-SHARED-EXPORTS:     "-b32"
// CHECK-LD32-SHARED-EXPORTS:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-SHARED-EXPORTS-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-SHARED-EXPORTS-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-SHARED-EXPORTS:     "-b{{(" ")?}}E:input.exp"
// CHECK-LD32-SHARED-EXPORTS-NOT: "-bE:{{[^"]+}}"
// CHECK-LD32-SHARED-EXPORTS:     "-lc++"
// CHECK-LD32-SHARED-EXPORTS:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-SHARED-EXPORTS:     "-lm"
// CHECK-LD32-SHARED-EXPORTS:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED %s
// CHECK-LD64-SHARED:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-SHARED:     "-isysroot" "[[SYSROOT:[^"]+]]"
//CHECK-LD64-SHARED:     "{{.*}}llvm-nm"
// CHECK-LD64-SHARED:     "--export-symbols"
// CHECK-LD64-SHARED:     "-X" "64"
// CHECK-LD64-SHARED:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED:     "-bM:SRE"
// CHECK-LD64-SHARED:     "-bnoentry"
// CHECK-LD64-SHARED:     "-b64"
// CHECK-LD64-SHARED:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED:     "-lc++"
// CHECK-LD64-SHARED:     "-lc++abi"
// CHECK-LD64-SHARED:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-SHARED-NOT: "--as-needed"
// CHECK-LD64-SHARED:     "-lunwind"
// CHECK-LD64-SHARED-NOT: "--no-as-needed"
// CHECK-LD64-SHARED:     "-lm"
// CHECK-LD64-SHARED:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -fno-exceptions.
// RUN: %clangxx %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -fno-exceptions \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOEXCEPTIONS %s
// CHECK-LD32-NOEXCEPTIONS:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOEXCEPTIONS:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD32-NOEXCEPTIONS:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOEXCEPTIONS:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOEXCEPTIONS:     "-b32"
// CHECK-LD32-NOEXCEPTIONS:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOEXCEPTIONS:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOEXCEPTIONS:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOEXCEPTIONS:     "-lc++"
// CHECK-LD32-NOEXCEPTIONS:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-LD32-NOEXCEPTIONS:     "-lm"
// CHECK-LD32-NOEXCEPTIONS:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with export list.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Wl,-bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with export list (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with export list (-Xlinker variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Xlinker -bE:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS %s

// CHECK-LD64-SHARED-EXPORTS:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED-EXPORTS:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-SHARED-EXPORTS:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-SHARED-EXPORTS-NOT: "{{.*}}llvm-nm"
// CHECK-LD64-SHARED-EXPORTS-NOT: "-X"
// CHECK-LD64-SHARED-EXPORTS-NOT: "64"
// CHECK-LD64-SHARED-EXPORTS:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED-EXPORTS:     "-bM:SRE"
// CHECK-LD64-SHARED-EXPORTS:     "-bnoentry"
// CHECK-LD64-SHARED-EXPORTS:     "-b64"
// CHECK-LD64-SHARED-EXPORTS:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-EXPORTS-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-EXPORTS-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED-EXPORTS:     "-b{{(" ")?}}E:input.exp"
// CHECK-LD64-SHARED-EXPORTS-NOT: "-bE:{{[^"]+}}"
// CHECK-LD64-SHARED-EXPORTS:     "-lc++"
// CHECK-LD64-SHARED-EXPORTS:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-SHARED-EXPORTS:     "-lm"
// CHECK-LD64-SHARED-EXPORTS:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with alternate export list.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Wl,-bexport:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS-ALT %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with alternate export list (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -bexport:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS-ALT %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with alternate export list (-Xlinker variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Xlinker -bexport:input.exp \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPORTS-ALT %s

// CHECK-LD64-SHARED-EXPORTS-ALT:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "{{.*}}llvm-nm"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "-X"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "64"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-bM:SRE"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-bnoentry"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-b64"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-b{{(" ")?}}export:input.exp"
// CHECK-LD64-SHARED-EXPORTS-ALT-NOT: "-bE:{{[^"]+}}"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-lc++"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-lm"
// CHECK-LD64-SHARED-EXPORTS-ALT:     "-lc"

// Check powerpc-ibm-aix7.3.0.0, -fprofile-generate
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -static \
// RUN:        -fprofile-generate\
// RUN:        --target=powerpc-ibm-aix7.3.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-PGO-NON-LTO %s
// CHECK-PGO-NON-LTO-NOT: warning:
// CHECK-PGO-NON-LTO:     "-cc1" "-triple" "powerpc-ibm-aix7.3.0.0"
// CHECK-PGO-NON-LTO:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-PGO-NON-LTO:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-PGO-NON-LTO:     "{{.*}}ld{{(.exe)?}}"
// CHECK-PGO-NON-LTO:     "-bdbg:namedsects:ss"
// CHECK-PGO-NON-LTO:     "-b32"
// CHECK-PGO-NON-LTO:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-PGO-NON-LTO:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-PGO-NON-LTO-NOT: "-lc++"
// CHECK-PGO-NON-LTO-NOT: "-lc++abi"
// CHECK-PGO-NON-LTO:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-PGO-NON-LTO-NOT: "--as-needed"
// CHECK-PGO-NON-LTO-NOT: "-lunwind"
// CHECK-PGO-NON-LTO-NOT: "--no-as-needed"
// CHECK-PGO-NON-LTO-NOT: "-lm"
// CHECK-PGO-NON-LTO:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpall.
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Wl,-bexpall \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPALL %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpall (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -bexpall \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPALL %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpall (-Xlinker variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Xlinker -bexpall \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPALL %s

// CHECK-LD64-SHARED-EXPALL:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED-EXPALL:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-SHARED-EXPALL:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-SHARED-EXPALL-NOT: "{{.*}}llvm-nm"
// CHECK-LD64-SHARED-EXPALL-NOT: "-X"
// CHECK-LD64-SHARED-EXPALL-NOT: "64"
// CHECK-LD64-SHARED-EXPALL:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED-EXPALL:     "-bM:SRE"
// CHECK-LD64-SHARED-EXPALL:     "-bnoentry"
// CHECK-LD64-SHARED-EXPALL:     "-b64"
// CHECK-LD64-SHARED-EXPALL:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-EXPALL-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-EXPALL-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED-EXPALL:     "-b{{(" ")?}}expall"
// CHECK-LD64-SHARED-EXPALL-NOT: "-bE:{{[^"]+}}"
// CHECK-LD64-SHARED-EXPALL:     "-lc++"
// CHECK-LD64-SHARED-EXPALL:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-SHARED-EXPALL:     "-lm"
// CHECK-LD64-SHARED-EXPALL:     "-lc"

// Check powerpc-ibm-aix7.2.5.3, -fprofile-generate, -flto
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        -static \
// RUN:        -fprofile-generate\
// RUN:        -flto\
// RUN:        --target=powerpc-ibm-aix7.2.5.3 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:   | FileCheck --check-prefix=CHECK-PGO-LTO %s
// CHECK-PGO-LTO-NOT: warning:
// CHECK-PGO-LTO:     "-cc1" "-triple" "powerpc-ibm-aix7.2.5.3"
// CHECK-PGO-LTO:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-PGO-LTO:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-PGO-LTO:     "{{.*}}ld{{(.exe)?}}"
// CHECK-PGO-LTO:     "-bdbg:namedsects:ss"
// CHECK-PGO-LTO:     "-b32"
// CHECK-PGO-LTO:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-PGO-LTO:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-PGO-LTO-NOT: "-lc++"
// CHECK-PGO-LTO-NOT: "-lc++abi"
// CHECK-PGO-LTO:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-PGO-LTO-NOT: "--as-needed"
// CHECK-PGO-LTO-NOT: "-lunwind"
// CHECK-PGO-LTO-NOT: "--no-as-needed"
// CHECK-PGO-LTO-NOT: "-lm"
// CHECK-PGO-LTO:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpfull (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Wl,-bexpfull \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPFULL %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpfull (no -Wl, variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -bexpfull \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPFULL %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared with -bexpfull (-Xlinker variant).
// RUN: %clangxx -x c++ %s 2>&1 -### \
// RUN:        -resource-dir=%S/Inputs/resource_dir -shared \
// RUN:        --target=powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -Xlinker -bexpfull \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED-EXPFULL %s

// CHECK-LD64-SHARED-EXPFULL:     "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED-EXPFULL:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-LD64-SHARED-EXPFULL:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-SHARED-EXPFULL-NOT: "{{.*}}llvm-nm"
// CHECK-LD64-SHARED-EXPFULL-NOT: "-X"
// CHECK-LD64-SHARED-EXPFULL-NOT: "64"
// CHECK-LD64-SHARED-EXPFULL:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED-EXPFULL:     "-bM:SRE"
// CHECK-LD64-SHARED-EXPFULL:     "-bnoentry"
// CHECK-LD64-SHARED-EXPFULL:     "-b64"
// CHECK-LD64-SHARED-EXPFULL:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-EXPFULL-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-EXPFULL-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED-EXPFULL:     "-b{{(" ")?}}expfull"
// CHECK-LD64-SHARED-EXPFULL-NOT: "-bE:{{[^"]+}}"
// CHECK-LD64-SHARED-EXPFULL:     "-lc++"
// CHECK-LD64-SHARED-EXPFULL:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc64.a"
// CHECK-LD64-SHARED-EXPFULL:     "-lm"
// CHECK-LD64-SHARED-EXPFULL:     "-lc"

// Check powerpc-ibm-aix7.1.0.0. -fopenmp=libomp to specify libomp explicitly.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -fopenmp=libomp \
// RUN:   | FileCheck --check-prefixes=CHECK-FOPENMP,CHECK-FOPENMP-OMP %s

// Check powerpc-ibm-aix7.1.0.0. -fopenmp=libiomp5 to specify libgomp explicitly.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -fopenmp=libiomp5 \
// RUN:   | FileCheck --check-prefixes=CHECK-FOPENMP,CHECK-FOPENMP-IOMP5 %s

// Check powerpc-ibm-aix7.1.0.0. -fopenmp=libgomp to specify libgomp explicitly.
// RUN: %clang %s -### 2>&1 \
// RUN:        -resource-dir=%S/Inputs/resource_dir \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -fopenmp=libgomp \
// RUN:   | FileCheck --check-prefixes=CHECK-FOPENMP,CHECK-FOPENMP-GOMP %s

// CHECK-FOPENMP-NOT: warning:
// CHECK-FOPENMP:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-FOPENMP:     "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-FOPENMP:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-FOPENMP:     "{{.*}}ld{{(.exe)?}}"
// CHECK-FOPENMP-NOT: "-bnso"
// CHECK-FOPENMP:     "-b32"
// CHECK-FOPENMP:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-FOPENMP:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-FOPENMP:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-FOPENMP-NOT: "-lc++"
// CHECK-FOPENMP-NOT: "-lc++abi"
// CHECK-FOPENMP:     "[[RESOURCE_DIR]]{{/|\\\\}}lib{{/|\\\\}}aix{{/|\\\\}}libclang_rt.builtins-powerpc.a"
// CHECK-FOPENMP-NOT: "--as-needed"
// CHECK-FOPENMP:     "-lunwind"
// CHECK-FOPENMP-NOT: "--no-as-needed"
// CHECK-FOPENMP-NOT: "-lm"
// CHECK-FOPENMP-OMP:     "-lomp"
// CHECK-FOPENMP-IOMP5:   "-liomp5"
// CHECK-FOPENMP-GOMP:    "-lgomp"
// CHECK-FOPENMP:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -fopenmp=libfoo results an error.
// RUN: not %clang %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        -fopenmp=libfoo \
// RUN:   | FileCheck --check-prefixes=CHECK-FOPENMP-FOO %s
// CHECK-FOPENMP-FOO: error: unsupported argument 'libfoo' to option '-fopenmp='

// Check powerpc-ibm-aix7.1.0.0. -r does not link object files or libraries
// RUN: %clang %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -L/foo/bar \
// RUN:        -r \
// RUN:   | FileCheck --check-prefixes=CHECK-RELOCATABLE %s

// CHECK-RELOCATABLE:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-RELOCATABLE:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-RELOCATABLE:     "{{.*}}ld{{(.exe)?}}"
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE:     "-L/foo/bar"
// CHECK-RELOCATABLE-NOT:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-RELOCATABLE-NOT:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-RELOCATABLE-NOT:     "-l{{.*}}"
// CHECK-RELOCATABLE-NOT:     "-L{{.*}}"

// Check powerpc-ibm-aix7.1.0.0. -K is a passthrough linker option.
// RUN: %clang %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -K \
// RUN:   | FileCheck --check-prefixes=CHECK-K %s
// CHECK-K:     "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-K:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-K:     "{{.*}}ld{{(.exe)?}}"
// CHECK-K:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-K:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-K:     "-K"

// Check powerpc-ibm-aix7.1.0.0. -K unused when not linking.
// RUN: %clang %s 2>&1 -### \
// RUN:        --target=powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:        --unwindlib=libunwind \
// RUN:        -K \
// RUN:        -c \
// RUN:   | FileCheck --check-prefixes=CHECK-K-UNUSED %s
// CHECK-K-UNUSED: clang: warning: -K: 'linker' input unused [-Wunused-command-line-argument]
