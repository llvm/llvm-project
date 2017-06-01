// RUN: %clang -target i386--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=I386 %s
// RUN: %clang -target x86_64--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=I386 %s
// I386: "-cc1" "-triple" "i386--netbsd"

// RUN: %clang -target i386--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=X86_64 %s
// RUN: %clang -target x86_64--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=X86_64 %s
// X86_64: "-cc1" "-triple" "x86_64--netbsd"

// r196538 set arm1176jzf-s as default CPU for ARMv6 on NetBSD
// RUN: %clang -target armv6--netbsd-eabihf -m32 %s -### 2>&1 | FileCheck -check-prefix=ARMV6 %s
// ARMV6: "-cc1" "-triple" "armv6kz--netbsd-eabihf"

// RUN: %clang -target sparcv9--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=SPARC %s
// RUN: %clang -target sparc--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=SPARC %s
// SPARC: "-cc1" "-triple" "sparc--netbsd"

// RUN: %clang -target sparcv9--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=SPARCV9 %s
// RUN: %clang -target sparc--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=SPARCV9 %s
// SPARCV9: "-cc1" "-triple" "sparcv9--netbsd"

// RUN: %clang -target sparc64--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=SPARC64 %s
// SPARC64: "-cc1" "-triple" "sparc64--netbsd"

// RUN: %clang -target sparcel -o foo %s -### 2>&1 | FileCheck -check-prefix=SPARCEL %s
// SPARCEL: gcc{{(\.exe)?}}" "-EL" "-o" "foo"

// RUN: %clang -target mips64--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=MIPS %s
// RUN: %clang -target mips--netbsd -m32 %s -### 2>&1 | FileCheck -check-prefix=MIPS %s
// MIPS: "-cc1" "-triple" "mips--netbsd"

// RUN: %clang -target mips64--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=MIPS64 %s
// RUN: %clang -target mips--netbsd -m64 %s -### 2>&1 | FileCheck -check-prefix=MIPS64 %s
// MIPS64: "-cc1" "-triple" "mips64--netbsd"
