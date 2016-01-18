// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd6.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-6 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM %s
// RUN: %clang -no-canonical-prefixes -target armeb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARMEB %s
// RUN: %clang -no-canonical-prefixes -target armeb--netbsd-eabi -march=armv7 \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARMV7EB %s
// RUN: %clang -no-canonical-prefixes -target armv7eb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARMV7EB %s
// RUN: %clang -r -no-canonical-prefixes -target armeb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARMEB-R %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-APCS %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabihf \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-HF %s
// RUN: %clang -no-canonical-prefixes -target thumb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=THUMB %s
// RUN: %clang -no-canonical-prefixes -target thumbeb--netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=THUMBEB %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd7.0.0-eabi \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd6.0.0-eabi \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-6 %s
// RUN: %clang -no-canonical-prefixes -target sparc--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s
// RUN: %clang -no-canonical-prefixes -target sparc64--netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64 %s
// RUN: %clang -no-canonical-prefixes -target powerpc--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC %s
// RUN: %clang -no-canonical-prefixes -target powerpc64--netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC64 %s

// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clang -no-canonical-prefixes -target x86_64--netbsd6.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-6 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64 %s
// RUN: %clang -no-canonical-prefixes -target aarch64--netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM %s
// RUN: %clang -no-canonical-prefixes -target armeb--netbsd-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARMEB %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd7.0.0-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-7 %s
// RUN: %clang -no-canonical-prefixes -target arm--netbsd6.0.0-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-6 %s
// RUN: %clang -no-canonical-prefixes -target sparc--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC %s
// RUN: %clang -no-canonical-prefixes -target sparc64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64 %s
// RUN: %clang -no-canonical-prefixes -target powerpc--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC %s
// RUN: %clang -no-canonical-prefixes -target powerpc64--netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC64 %s

// X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd"
// AARCH64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd7.0.0"
// AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// ARM: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARM: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM: "-m" "armelf_nbsd_eabi"
// ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARMEB: clang{{.*}}" "-cc1" "-triple" "armebv5e--netbsd-eabi"
// ARMEB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARMEB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARMEB-NOT: "--be8"
// ARMEB: "-m" "armelfb_nbsd_eabi"
// ARMEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
// ARMV7EB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARMV7EB: "--be8" "-m" "armelfb_nbsd_eabi"

// ARMEB-R: ld{{.*}}"
// ARMEB-R-NOT: "--be8"

// ARM-APCS: clang{{.*}}" "-cc1" "-triple" "armv4--netbsd"
// ARM-APCS: as{{.*}}" "-mcpu=strongarm" "-o"
// ARM-APCS: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-APCS: "-m" "armelf_nbsd"
// ARM-APCS: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}oabi{{/|\\\\}}crti.o"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-HF: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabihf"
// ARM-HF: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARM-HF: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-HF: "-m" "armelf_nbsd_eabihf"
// ARM-HF: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}eabihf{{/|\\\\}}crti.o"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// THUMB: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// THUMB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// THUMB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// THUMB: "-m" "armelf_nbsd_eabi"
// THUMB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// THUMBEB: clang{{.*}}" "-cc1" "-triple" "armebv5e--netbsd-eabi"
// THUMBEB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// THUMBEB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// THUMBEB: "-m" "armelfb_nbsd_eabi"
// THUMBEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd7.0.0-eabi"
// ARM-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-7: "-m" "armelf_nbsd_eabi"
// ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-6: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd6.0.0-eabi"
// ARM-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-6: "-m" "armelf_nbsd_eabi"
// ARM-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-6: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC: clang{{.*}}" "-cc1" "-triple" "sparc--netbsd"
// SPARC: as{{.*}}" "-32" "-o"
// SPARC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC: "-m" "elf32_sparc"
// SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64--netbsd"
// SPARC64: as{{.*}}" "-64" "-Av9" "-o"
// SPARC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64: "-m" "elf64_sparc"
// SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC: clang{{.*}}" "-cc1" "-triple" "powerpc--netbsd"
// POWERPC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC: "-m" "elf32ppc_nbsd"
// POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC64: clang{{.*}}" "-cc1" "-triple" "powerpc64--netbsd"
// POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC64: "-m" "elf64ppc"
// POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd"
// S-X86_64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64--netbsd6.0.0"
// S-X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64-6: "-lgcc_eh" "-lc" "-lgcc"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd"
// S-AARCH64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64--netbsd7.0.0"
// S-AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd-eabi"
// S-ARM: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM: "-m" "armelf_nbsd_eabi"
// S-ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARMEB: clang{{.*}}" "-cc1" "-triple" "armebv5e--netbsd-eabi"
// S-ARMEB: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARMEB: "-m" "armelfb_nbsd_eabi"
// S-ARMEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd7.0.0-eabi"
// S-ARM-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-7: "-m" "armelf_nbsd_eabi"
// S-ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-6: clang{{.*}}" "-cc1" "-triple" "armv5e--netbsd6.0.0-eabi"
// S-ARM-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-6: "-m" "armelf_nbsd_eabi"
// S-ARM-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARM-6: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM-6: "-lgcc_eh" "-lc" "-lgcc"
// S-ARM-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC: clang{{.*}}" "-cc1" "-triple" "sparc--netbsd"
// S-SPARC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC: "-m" "elf32_sparc"
// S-SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC: "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64--netbsd"
// S-SPARC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64: "-m" "elf64_sparc"
// S-SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC64: "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC: clang{{.*}}" "-cc1" "-triple" "powerpc--netbsd"
// S-POWERPC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC: "-m" "elf32ppc_nbsd"
// S-POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC64: clang{{.*}}" "-cc1" "-triple" "powerpc64--netbsd"
// S-POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC64: "-m" "elf64ppc"
// S-POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
