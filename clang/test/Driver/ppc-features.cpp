/// Check default CC1 and linker options for ppc32.
// RUN: %clang -### -target powerpcle-unknown-linux-gnu %s 2>&1 | FileCheck --check-prefixes=PPC32,PPC32LELNX %s
// RUN: %clang -### -target powerpc-unknown-linux-gnu %s 2>&1 | FileCheck --check-prefixes=PPC32,PPC32BELNX %s
// RUN: %clang -### -target powerpcle-unknown-freebsd13.0 %s 2>&1 | FileCheck --check-prefixes=PPC32,PPC32LEFBSD %s
// RUN: %clang -### -target powerpc-unknown-freebsd13.0 %s 2>&1 | FileCheck --check-prefixes=PPC32,PPC32BEFBSD %s
// PPC32:      "-funwind-tables=2"
// PPC32-SAME: "-mfloat-abi" "hard"

// PPC32LELNX-NEXT: "-m" "elf32lppclinux"
// PPC32BELNX-NEXT: "-m" "elf32ppclinux"
// PPC32LEFBSD-NEXT: "-m" "elf32lppc"
// PPC32BEFBSD-NEXT: "-m" "elf32ppc_fbsd"

// check -msoft-float option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -msoft-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-SOFTFLOAT %s
// CHECK-SOFTFLOAT: "-target-feature" "-hard-float"

// check -mfloat-abi=soft option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -mfloat-abi=soft -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-FLOATABISOFT %s
// CHECK-FLOATABISOFT: "-target-feature" "-hard-float"

// check -mhard-float option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -mhard-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-HARDFLOAT %s
// CHECK-HARDFLOAT-NOT: "-target-feature" "-hard-float"

// check -mfloat-abi=hard option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -mfloat-abi=hard -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-FLOATABIHARD %s
// CHECK-FLOATABIHARD-NOT: "-target-feature" "-hard-float"

// check combine -mhard-float -msoft-float option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -mhard-float -msoft-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-HARDSOFT %s
// CHECK-HARDSOFT: "-target-feature" "-hard-float"

// check combine -msoft-float -mhard-float option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu %s -msoft-float -mhard-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-SOFTHARD %s
// CHECK-SOFTHARD-NOT: "-target-feature" "-hard-float"

// check -msecure-plt option for ppc32
// RUN: %clang -target powerpc-unknown-linux-gnu -msecure-plt %s  -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-SECUREPLT %s
// CHECK-SECUREPLT: "-target-feature" "+secure-plt"

// check -mfloat-abi=x option
// RUN: not %clang --target=powerpc-unknown-linux-gnu %s -mfloat-abi=x -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-ERRMSG %s
// CHECK-ERRMSG: error: invalid float ABI '-mfloat-abi=x'


/// Check default CC1 and linker options for ppc64.
// RUN: %clang -### -target powerpc64le-unknown-linux-gnu %s 2>&1 | FileCheck --check-prefixes=PPC64,PPC64LE %s
// RUN: %clang -### -target powerpc64-unknown-linux-gnu %s 2>&1 | FileCheck --check-prefixes=PPC64,PPC64BE %s
// PPC64:      "-funwind-tables=2"
// PPC64-SAME: "-mfloat-abi" "hard"

// PPC64LE: "-m" "elf64lppc"
// PPC64BE: "-m" "elf64ppc"

// check -msoft-float option for ppc64
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -msoft-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-SOFTFLOAT64 %s
// CHECK-SOFTFLOAT64: "-target-feature" "-hard-float"

// check -mfloat-abi=soft option for ppc64
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mfloat-abi=soft -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-FLOATABISOFT64 %s
// CHECK-FLOATABISOFT64: "-target-feature" "-hard-float"

// check -msoft-float option for ppc64
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -msoft-float -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-SOFTFLOAT64le %s
// CHECK-SOFTFLOAT64le: "-target-feature" "-hard-float"

// check -mfloat-abi=soft option for ppc64
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -mfloat-abi=soft -### -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-FLOATABISOFT64le %s
// CHECK-FLOATABISOFT64le: "-target-feature" "-hard-float"

// Check that -mno-altivec correctly disables the altivec target feature on powerpc.

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-altivec -maltivec -### -o %t.o 2>&1 | FileCheck --check-prefix=ALTIVEC %s
// ALTIVEC: "-target-feature" "+altivec"

// RUN: %clang -### -c -target powerpc64-unknown-linux-gnu %s -maltivec -mno-altivec 2>&1 | FileCheck --check-prefix=NO_ALTIVEC %s
// NO_ALTIVEC: "-target-feature" "-altivec"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-mfcrf -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOMFCRF %s
// CHECK-NOMFCRF: "-target-feature" "-mfocrf"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-mfcrf -mmfcrf -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-MFCRF %s
// CHECK-MFCRF: "-target-feature" "+mfocrf"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-isel -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOISEL %s
// CHECK-NOISEL: "-target-feature" "-isel"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-isel -misel -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-ISEL %s
// CHECK-ISEL: "-target-feature" "+isel"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-popcntd -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOPOPCNTD %s
// CHECK-NOPOPCNTD: "-target-feature" "-popcntd"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-popcntd -mpopcntd -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-POPCNTD %s
// CHECK-POPCNTD: "-target-feature" "+popcntd"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-fprnd -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOFPRND %s
// CHECK-NOFPRND: "-target-feature" "-fprnd"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-fprnd -mfprnd -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-FPRND %s
// CHECK-FPRND: "-target-feature" "+fprnd"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-cmpb -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCMPB %s
// CHECK-NOCMPB: "-target-feature" "-cmpb"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-cmpb -mcmpb -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CMPB %s
// CHECK-CMPB: "-target-feature" "+cmpb"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-vsx -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOVSX %s
// CHECK-NOVSX: "-target-feature" "-vsx"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-vsx -mvsx -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-VSX %s
// CHECK-VSX: "-target-feature" "+vsx"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-htm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOHTM %s
// CHECK-NOHTM: "-target-feature" "-htm"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-htm -mhtm -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-HTM %s
// CHECK-HTM: "-target-feature" "+htm"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-power8-vector -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOP8VECTOR %s
// CHECK-NOP8VECTOR: "-target-feature" "-power8-vector"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-power8-vector -mpower8-vector -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-P8VECTOR %s
// CHECK-P8VECTOR: "-target-feature" "+power8-vector"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-power10-vector -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOP10VECTOR %s
// CHECK-NOP10VECTOR: "-target-feature" "-power10-vector"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-power10-vector -mpower10-vector -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-P10VECTOR %s
// CHECK-P10VECTOR: "-target-feature" "+power10-vector"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-crbits -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOCRBITS %s
// CHECK-NOCRBITS: "-target-feature" "-crbits"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-crbits -mcrbits -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-CRBITS %s
// CHECK-CRBITS: "-target-feature" "+crbits"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-longcall -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOLONGCALL %s
// CHECK-NOLONGCALL: "-target-feature" "-longcall"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-longcall -mlongcall -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-LONGCALL %s
// CHECK-LONGCALL: "-target-feature" "+longcall"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-invariant-function-descriptors -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-NOINVFUNCDESC %s
// CHECK-NOINVFUNCDESC: "-target-feature" "-invariant-function-descriptors"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -mno-invariant-function-descriptors -minvariant-function-descriptors -### -o %t.o 2>&1 | FileCheck -check-prefix=CHECK-INVFUNCDESC %s
// CHECK-INVFUNCDESC: "-target-feature" "+invariant-function-descriptors"

// RUN: %clang -target powerpc %s -mno-spe -mspe -c -### 2>&1 | FileCheck -check-prefix=CHECK-SPE %s
// RUN: %clang -target powerpcspe %s -c -### 2>&1 | FileCheck -check-prefix=CHECK-SPE %s
// RUN: %clang -target powerpcspe %s -mno-spe -c -### 2>&1 | FileCheck -check-prefix=CHECK-NOSPE %s
// CHECK-SPE: "-target-feature" "+spe"
// CHECK-NOSPE: "-target-feature" "-spe"

// RUN: %clang -target powerpc %s -mefpu2 -c -### 2>&1 | FileCheck -check-prefix=CHECK-EFPU2 %s
// CHECK-EFPU2: "-target-feature" "+efpu2"

// Assembler features
// RUN: %clang -target powerpc-unknown-linux-gnu %s -### -o %t.o -no-integrated-as 2>&1 | FileCheck -check-prefix=CHECK_32_BE_AS_ARGS %s
// CHECK_32_BE_AS_ARGS: "-mppc"
// CHECK_32_BE_AS_ARGS-SAME: "-mbig-endian"
// CHECK_32_BE_AS_ARGS-SAME: "-many"

// RUN: %clang -target powerpcle-unknown-linux-gnu %s -### -o %t.o -no-integrated-as 2>&1 | FileCheck -check-prefix=CHECK_32_LE_AS_ARGS %s
// CHECK_32_LE_AS_ARGS: "-mppc"
// CHECK_32_LE_AS_ARGS-SAME: "-mlittle-endian"
// CHECK_32_LE_AS_ARGS-SAME: "-many"

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o -no-integrated-as 2>&1 | FileCheck -check-prefix=CHECK_64_BE_AS_ARGS %s
// CHECK_64_BE_AS_ARGS: "-mppc64"
// CHECK_64_BE_AS_ARGS-SAME: "-mbig-endian"
// CHECK_64_BE_AS_ARGS-SAME: "-many"

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o -no-integrated-as 2>&1 | FileCheck -check-prefix=CHECK_64_LE_AS_ARGS %s
// CHECK_64_LE_AS_ARGS: "-mppc64"
// CHECK_64_LE_AS_ARGS-SAME: "-mlittle-endian"
// CHECK_64_LE_AS_ARGS-SAME: "-mpower8"

// OpenMP features
// RUN: %clang -target powerpc-unknown-linux-gnu %s -### -fopenmp=libomp -o %t.o 2>&1 | FileCheck -check-prefix=CHECK_OPENMP_TLS %s
// RUN: %clang -target powerpcle-unknown-linux-gnu %s -### -fopenmp=libomp -o %t.o 2>&1 | FileCheck -check-prefix=CHECK_OPENMP_TLS %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -fopenmp=libomp -o %t.o 2>&1 | FileCheck -check-prefix=CHECK_OPENMP_TLS %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -fopenmp=libomp -o %t.o 2>&1 | FileCheck -check-prefix=CHECK_OPENMP_TLS %s
// CHECK_OPENMP_TLS-NOT: "-fnoopenmp-use-tls"
