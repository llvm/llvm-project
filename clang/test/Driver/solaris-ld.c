// General tests that ld invocations on Solaris targets sane. Note that we use
// sysroot to make these tests independent of the host system.

// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang -### %s 2>&1 --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32 %s
// CHECK-LD-SPARC32-NOT: warning:
// CHECK-LD-SPARC32: "-cc1" "-triple" "sparc-sun-solaris2.11"
// CHECK-LD-SPARC32-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-SPARC32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crt1.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtbegin.o"
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2"
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../.."
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-SPARC32-SAME: "-zignore" "-latomic" "-zrecord"
// CHECK-LD-SPARC32-SAME: "-lgcc_s"
// CHECK-LD-SPARC32-SAME: "-lc"
// CHECK-LD-SPARC32-SAME: "-lgcc"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtend.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crtn.o"

// Check sparc-sun-solaris2.11, 64bit
// RUN: %clang -m64 -### %s 2>&1 --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC64 %s
// CHECK-LD-SPARC64-NOT: warning:
// CHECK-LD-SPARC64: "-cc1" "-triple" "sparcv9-sun-solaris2.11"
// CHECK-LD-SPARC64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-SPARC64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crt1.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/lib/sparcv9{{/|\\\\}}crti.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crtbegin.o"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../sparcv9"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/lib/sparcv9"
// CHECK-LD-SPARC64-NOT:  "-latomic"
// CHECK-LD-SPARC64-SAME: "-lgcc_s"
// CHECK-LD-SPARC64-SAME: "-lc"
// CHECK-LD-SPARC64-SAME: "-lgcc"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crtend.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/lib/sparcv9{{/|\\\\}}crtn.o"

// Check i386-pc-solaris2.11, 32bit
// RUN: %clang -### %s 2>&1 --target=i386-pc-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32 %s
// CHECK-LD-X32-NOT: warning:
// CHECK-LD-X32: "-cc1" "-triple" "i386-pc-solaris2.11"
// CHECK-LD-X32-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-X32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt1.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4{{/|\\\\}}crtbegin.o"
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4"
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../.."
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-X32-NOT:  "-latomic"
// CHECK-LD-X32-SAME: "-lgcc_s"
// CHECK-LD-X32-SAME: "-lc"
// CHECK-LD-X32-SAME: "-lgcc"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4{{/|\\\\}}crtend.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crtn.o"

// Check i386-pc-solaris2.11, 64bit
// RUN: %clang -m64 -### %s 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X64 %s
// CHECK-LD-X64-NOT: warning:
// CHECK-LD-X64: "-cc1" "-triple" "x86_64-pc-solaris2.11"
// CHECK-LD-X64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-X64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crt1.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crti.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64{{/|\\\\}}crtbegin.o"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../amd64"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/lib/amd64"
// CHECK-LD-X64-NOT:  "-latomic"
// CHECK-LD-X64-SAME: "-lgcc_s"
// CHECK-LD-X64-SAME: "-lc"
// CHECK-LD-X64-SAME: "-lgcc"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64{{/|\\\\}}crtend.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crtn.o"

// Check the right -l flags are present with -shared
// RUN: %clang -### %s -shared 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SPARC32-SHARED %s
// CHECK-SPARC32-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARC32-SHARED-SAME: "-lgcc_s"
// CHECK-SPARC32-SHARED-SAME: "-lc"
// CHECK-SPARC32-SHARED-NOT: "-lgcc"

// Check that libm is only linked with clang++.
// RUN: %clang -### %s --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOLIBM %s
// RUN: %clang -### %s -shared --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOLIBM %s
// RUN: %clangxx -### %s --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBM %s
// RUN: %clangxx -### %s -shared --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBM %s
// CHECK-LIBM: "-lm"
// CHECK-NOLIBM-NOT: "-lm"

// Check the right ld flags are present with -pie.
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -pie \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PIE %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -nopie \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOPIE %s

// Check that -shared/-r/-static disable PIE.
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -shared -pie \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOPIE %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -r -pie \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOPIE %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -static -pie \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOPIE %s

// CHECK-PIE: "-z" "type=pie"
// CHECK-NOPIE-NOT: "-z" "type=pie"

// -r suppresses default -l and crt*.o, values-*.o like -nostdlib.
// RUN: %clang -### %s --target=sparc-sun-solaris2.11 -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE:     "-L
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE-NOT: "-l
// CHECK-RELOCATABLE-NOT: /crt{{[^.]+}}.o
// CHECK-RELOCATABLE-NOT: /values-{{[^.]+}}.o

// Check that crt{begin,end}S.o is linked with -shared/-pie.
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTS %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -shared \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTS %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -nopie \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTS %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -pie \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTS %s
// CHECK-CRTS: crtbeginS.o
// CHECK-CRTS: crtendS.o
// CHECK-NOCRTS-NOT: crtbeginS.o
// CHECK-NOCRTS-NOT: crtendS.o

// Check that crtfastmath.o is linked with -ffast-math.

// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH-SPARC32 %s
// RUN: %clang --target=sparc-sun-solaris2.11 -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH-SPARC32 %s
// CHECK-CRTFASTMATH-SPARC32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-CRTFASTMATH-SPARC32: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtfastmath.o"
// CHECK-NOCRTFASTMATH-SPARC32-NOT: crtfastmath.o

// Check sparc-pc-solaris2.11, 64bit
// RUN: %clang -m64 --target=sparc-sun-solaris2.11 -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH-SPARC64 %s
// RUN: %clang -m64 --target=sparc-sun-solaris2.11 -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_sparc_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH-SPARC64 %s
// CHECK-CRTFASTMATH-SPARC64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-CRTFASTMATH-SPARC64: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crtfastmath.o"
// CHECK-NOCRTFASTMATH-SPARC64-NOT: crtfastmath.o

// Check i386-pc-solaris2.11, 32bit
// RUN: %clang --target=i386-pc-solaris2.11 -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH-X32 %s
// RUN: %clang --target=i386-pc-solaris2.11 -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH-X32 %s
// CHECK-CRTFASTMATH-X32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-CRTFASTMATH-X32: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4{{/|\\\\}}crtfastmath.o"
// CHECK-NOCRTFASTMATH-X32-NOT: crtfastmath.o

// Check i386-pc-solaris2.11, 64bit
// RUN: %clang -m64 --target=i386-pc-solaris2.11 -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH-X64 %s
// RUN: %clang -m64 --target=i386-pc-solaris2.11 -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/solaris_x86_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH-X64 %s
// CHECK-CRTFASTMATH-X64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-CRTFASTMATH-X64: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64{{/|\\\\}}crtfastmath.o"
// CHECK-NOCRTFASTMATH-X64-NOT: crtfastmath.o
