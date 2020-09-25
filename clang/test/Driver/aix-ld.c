// General tests that ld invocations on AIX targets are sane. Note that we use
// sysroot to make these tests independent of the host system.

// Check powerpc-ibm-aix7.1.0.0, 32-bit.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32 %s
// CHECK-LD32-NOT: warning:
// CHECK-LD32:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOT: "-bnso"
// CHECK-LD32:     "-b32"
// CHECK-LD32:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NOT: "-lc++"
// CHECK-LD32:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64 %s
// CHECK-LD64-NOT: warning:
// CHECK-LD64:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOT: "-bnso"
// CHECK-LD64:     "-b64"
// CHECK-LD64:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NOT: "-lc++"
// CHECK-LD64:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Enable POSIX thread support.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -pthread \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-PTHREAD %s
// CHECK-LD32-PTHREAD-NOT: warning:
// CHECK-LD32-PTHREAD:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-PTHREAD:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-PTHREAD:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-PTHREAD-NOT: "-bnso"
// CHECK-LD32-PTHREAD:     "-b32"
// CHECK-LD32-PTHREAD:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-PTHREAD-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-PTHREAD:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-PTHREAD-NOT: "-lc++"
// CHECK-LD32-PTHREAD:     "-lpthreads"
// CHECK-LD32-PTHREAD:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. POSIX thread alias.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -pthreads \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-PTHREAD %s
// CHECK-LD64-PTHREAD-NOT: warning:
// CHECK-LD64-PTHREAD:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-PTHREAD:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-PTHREAD:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-PTHREAD-NOT: "-bnso"
// CHECK-LD64-PTHREAD:     "-b64"
// CHECK-LD64-PTHREAD:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-PTHREAD:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-PTHREAD-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-PTHREAD:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-PTHREAD-NOT: "-lc++"
// CHECK-LD64-PTHREAD:     "-lpthreads"
// CHECK-LD64-PTHREAD:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Enable profiling.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -p \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-PROF %s
// CHECK-LD32-PROF-NOT: warning:
// CHECK-LD32-PROF:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-PROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-PROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-PROF-NOT: "-bnso"
// CHECK-LD32-PROF:     "-b32"
// CHECK-LD32-PROF:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-PROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}mcrt0.o"
// CHECK-LD32-PROF-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-PROF:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-PROF-NOT: "-lc++"
// CHECK-LD32-PROF:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. Enable g-profiling.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -pg \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-GPROF %s
// CHECK-LD64-GPROF-NOT: warning:
// CHECK-LD64-GPROF:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-GPROF:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-GPROF:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-GPROF-NOT: "-bnso"
// CHECK-LD64-GPROF:     "-b64"
// CHECK-LD64-GPROF:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-GPROF:     "[[SYSROOT]]/usr/lib{{/|\\\\}}gcrt0_64.o"
// CHECK-LD64-GPROF-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-GPROF:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-GPROF-NOT: "-lc++"
// CHECK-LD64-GPROF:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Static linking.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -static \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-STATIC %s
// CHECK-LD32-STATIC-NOT: warning:
// CHECK-LD32-STATIC:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-STATIC:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-STATIC:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-STATIC:     "-bnso"
// CHECK-LD32-STATIC:     "-b32"
// CHECK-LD32-STATIC:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-STATIC:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-STATIC-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-STATIC:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-STATIC-NOT: "-lc++"
// CHECK-LD32-STATIC:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. Library search path.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -L%S/Inputs/aix_ppc_tree/powerpc-ibm-aix7.1.0.0 \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-LIBP %s
// CHECK-LD32-LIBP-NOT: warning:
// CHECK-LD32-LIBP:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-LIBP:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-LIBP:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-LIBP-NOT: "-bnso"
// CHECK-LD32-LIBP:     "-b32"
// CHECK-LD32-LIBP:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-LIBP:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-LIBP-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-LIBP:     "-L[[SYSROOT]]/powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-LIBP:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-LIBP-NOT: "-lc++"
// CHECK-LD32-LIBP:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. nostdlib.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -nostdlib \
// RUN:        -pthread \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NO-STD-LIB %s
// CHECK-LD32-NO-STD-LIB-NOT: warning:
// CHECK-LD32-NO-STD-LIB:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NO-STD-LIB:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NO-STD-LIB:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NO-STD-LIB-NOT: "-bnso"
// CHECK-LD32-NO-STD-LIB:     "-b32"
// CHECK-LD32-NO-STD-LIB:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NO-STD-LIB-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NO-STD-LIB-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NO-STD-LIB:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NO-STD-LIB-NOT: "-lc++"
// CHECK-LD32-NO-STD-LIB-NOT: "-lpthreads"
// CHECK-LD32-NO-STD-LIB-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. nodefaultlibs.
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:        -nodefaultlibs \
// RUN:        -pthread \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NO-DEFAULT-LIBS %s
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: warning:
// CHECK-LD64-NO-DEFAULT-LIBS:     {{.*}}clang{{(.exe)?}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NO-DEFAULT-LIBS:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-bnso"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-b64"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NO-DEFAULT-LIBS:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NO-DEFAULT-LIBS:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lc++"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lpthreads"
// CHECK-LD64-NO-DEFAULT-LIBS-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. 'bcdtors' and argument order.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -Wl,-bnocdtors \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-CXX-ARG-ORDER %s
// CHECK-LD32-CXX-ARG-ORDER:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
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
// CHECK-LD32-CXX-ARG-ORDER:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. lc++ and lc order.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-CXX-ARG-LCXX %s
// CHECK-LD32-CXX-ARG-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-CXX-ARG-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-CXX-ARG-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-CXX-ARG-LCXX:     "-b32"
// CHECK-LD32-CXX-ARG-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-CXX-ARG-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-CXX-ARG-LCXX:     "-lc++"
// CHECK-LD32-CXX-ARG-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. lc++ and lc order.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-CXX-ARG-LCXX %s
// CHECK-LD64-CXX-ARG-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-CXX-ARG-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-CXX-ARG-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-CXX-ARG-LCXX:     "-b64"
// CHECK-LD64-CXX-ARG-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-CXX-ARG-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-CXX-ARG-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-CXX-ARG-LCXX:     "-lc++"
// CHECK-LD64-CXX-ARG-LCXX:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nodefaultlibs.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nodefaultlibs \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NODEFLIB-LCXX %s
// CHECK-LD32-NODEFLIB-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NODEFLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NODEFLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NODEFLIB-LCXX:     "-b32"
// CHECK-LD32-NODEFLIB-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NODEFLIB-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lc++"
// CHECK-LD32-NODEFLIB-LCXX-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nodefaultlibs.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nodefaultlibs \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NODEFLIB-LCXX %s
// CHECK-LD64-NODEFLIB-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NODEFLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NODEFLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NODEFLIB-LCXX:     "-b64"
// CHECK-LD64-NODEFLIB-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NODEFLIB-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NODEFLIB-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lc++"
// CHECK-LD64-NODEFLIB-LCXX-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nostdlib.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostdlib \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTDLIB-LCXX %s
// CHECK-LD32-NOSTDLIB-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTDLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTDLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTDLIB-LCXX:     "-b32"
// CHECK-LD32-NOSTDLIB-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTDLIB-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lc++"
// CHECK-LD32-NOSTDLIB-LCXX-NOT: "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostdlib.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostdlib \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NOSTDLIB-LCXX %s
// CHECK-LD64-NOSTDLIB-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTDLIB-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTDLIB-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTDLIB-LCXX:     "-b64"
// CHECK-LD64-NOSTDLIB-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTDLIB-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lc++"
// CHECK-LD64-NOSTDLIB-LCXX-NOT: "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -nostdlib++.
// Note: crti is still linked for initialization/finalization functionality.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostdlib++ \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTDLIBXX-LCXX %s
// CHECK-LD32-NOSTDLIBXX-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-b32"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NOSTDLIBXX-LCXX-NOT: "-lc++"
// CHECK-LD32-NOSTDLIBXX-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostdlib++.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostdlib++ \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-NOSTDLIBXX-LCXX %s
// CHECK-LD64-NOSTDLIBXX-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-b64"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NOSTDLIBXX-LCXX-NOT: "-lc++"
// CHECK-LD64-NOSTDLIBXX-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 32-bit. -nostartfiles.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostartfiles \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTARTFILES-LCXX %s
// CHECK-LD32-NOSTARTFILES-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-NOSTARTFILES-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-b32"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-NOSTARTFILES-LCXX      "-lc++"
// CHECK-LD32-NOSTARTFILES-LCXX:     "-lc"

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -nostartfiles.
// RUN: %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -nostartfiles \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-NOSTARTFILES-LCXX %s
// CHECK-LD64-NOSTARTFILES-LCXX:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-NOSTARTFILES-LCXX:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-b64"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-NOSTARTFILES-LCXX-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lc++"
// CHECK-LD64-NOSTARTFILES-LCXX:     "-lc"

// Check powerpc-ibm-aix7.1.0.0, 32-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        -stdlib=libstdc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-LIBSTDCXX %s

// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -stdlib=libstdc++ invokes fatal error.
// RUN: not --crash %clangxx -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        -stdlib=libstdc++ \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-LIBSTDCXX %s
// CHECK-LD-LIBSTDCXX: LLVM ERROR: linking libstdc++ unimplemented on AIX

// Check powerpc64-ibm-aix7.1.0.0, 32-bit. -shared.
// RUN: %clangxx -x c++ -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -shared \
// RUN:        -target powerpc-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD32-SHARED %s
// CHECK-LD32-SHARED:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc-ibm-aix7.1.0.0"
// CHECK-LD32-SHARED:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD32-SHARED:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD32-SHARED:     "-bM:SRE"
// CHECK-LD32-SHARED:     "-bnoentry"
// CHECK-LD32-SHARED:     "-b32"
// CHECK-LD32-SHARED:     "-bpT:0x10000000" "-bpD:0x20000000"
// CHECK-LD32-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0.o"
// CHECK-LD32-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD32-SHARED:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD32-SHARED:     "-lc++"
// CHECK-LD32-SHARED:     "-lc"


// Check powerpc64-ibm-aix7.1.0.0, 64-bit. -shared.
// RUN: %clangxx -x c++ -no-canonical-prefixes %s 2>&1 -### \
// RUN:        -shared \
// RUN:        -target powerpc64-ibm-aix7.1.0.0 \
// RUN:        --sysroot %S/Inputs/aix_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD64-SHARED %s
// CHECK-LD64-SHARED:     {{.*}}clang{{.*}}" "-cc1" "-triple" "powerpc64-ibm-aix7.1.0.0"
// CHECK-LD64-SHARED:     "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD64-SHARED:     "{{.*}}ld{{(.exe)?}}"
// CHECK-LD64-SHARED:     "-bM:SRE"
// CHECK-LD64-SHARED:     "-bnoentry"
// CHECK-LD64-SHARED:     "-b64"
// CHECK-LD64-SHARED:     "-bpT:0x100000000" "-bpD:0x110000000"
// CHECK-LD64-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt0_64.o"
// CHECK-LD64-SHARED-NOT: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti_64.o"
// CHECK-LD64-SHARED:     "-L[[SYSROOT]]/usr/lib"
// CHECK-LD64-SHARED:     "-lc++"
// CHECK-LD64-SHARED:     "-lc"
