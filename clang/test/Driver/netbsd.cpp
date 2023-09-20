// RUN: %clangxx --target=x86_64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clangxx --target=x86_64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clangxx --target=arm-unknown-netbsd7.0.0-eabi \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-7 %s
// RUN: %clangxx --target=aarch64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64 %s
// RUN: %clangxx --target=aarch64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64-7 %s
// RUN: %clangxx --target=aarch64_be-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE %s
// RUN: %clangxx --target=aarch64_be-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE-7 %s
// RUN: %clangxx --target=sparc-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s
// RUN: %clangxx --target=sparc-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-7 %s
// RUN: %clangxx --target=sparc64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64 %s
// RUN: %clangxx --target=sparc64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64-7 %s
// RUN: %clangxx --target=powerpc-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC %s
// RUN: %clangxx --target=powerpc64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC64 %s

// RUN: %clangxx --target=x86_64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clangxx --target=x86_64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clangxx --target=arm-unknown-netbsd7.0.0-eabi -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-7 %s
// RUN: %clangxx --target=aarch64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64 %s
// RUN: %clangxx --target=aarch64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64-7 %s
// RUN: %clangxx --target=aarch64_be-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE %s
// RUN: %clangxx --target=aarch64_be-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE-7 %s
// RUN: %clangxx --target=sparc-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC %s
// RUN: %clangxx --target=sparc-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC-7 %s
// RUN: %clangxx --target=sparc64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64 %s
// RUN: %clangxx --target=sparc64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64-7 %s
// RUN: %clangxx --target=powerpc-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC %s
// RUN: %clangxx --target=powerpc64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC64 %s

// X86_64: "-cc1" "-triple" "x86_64-unknown-netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-7: "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// ARM-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++" "-lm" "-lc"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64: "-cc1" "-triple" "aarch64-unknown-netbsd"
// AARCH64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64: "-lm" "-lc"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64-7: "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64-7: "-lm" "-lc"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE: "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64_BE: "-lm" "-lc"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE-7: "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64_BE-7: "-lm" "-lc"
// AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC: "-cc1" "-triple" "sparc-unknown-netbsd"
// SPARC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC: "-lm" "-lc"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC-7: "-cc1" "-triple" "sparc-unknown-netbsd7.0.0"
// SPARC-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC-7: "-lm" "-lc"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64: "-cc1" "-triple" "sparc64-unknown-netbsd"
// SPARC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC64: "-lm" "-lc"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64-7: "-cc1" "-triple" "sparc64-unknown-netbsd7.0.0"
// SPARC64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC64-7: "-lm" "-lc"
// SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC: "-cc1" "-triple" "powerpc-unknown-netbsd"
// POWERPC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// POWERPC: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC64: "-cc1" "-triple" "powerpc64-unknown-netbsd"
// POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// POWERPC64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: "-cc1" "-triple" "x86_64-unknown-netbsd"
// S-X86_64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-7: "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// S-ARM-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++" "-lm" "-lc"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64: "-cc1" "-triple" "aarch64-unknown-netbsd"
// S-AARCH64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64: "-lm" "-lc"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64-7: "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// S-AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64-7: "-lm" "-lc"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE: "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// S-AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64_BE: "-lm" "-lc"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE-7: "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// S-AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64_BE-7: "-lm" "-lc"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC: "-cc1" "-triple" "sparc-unknown-netbsd"
// S-SPARC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC: "-lm" "-lc"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC-7: "-cc1" "-triple" "sparc-unknown-netbsd7.0.0"
// S-SPARC-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC-7: "-lm" "-lc"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64: "-cc1" "-triple" "sparc64-unknown-netbsd"
// S-SPARC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC64: "-lm" "-lc"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64-7: "-cc1" "-triple" "sparc64-unknown-netbsd7.0.0"
// S-SPARC64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC64-7: "-lm" "-lc"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC: "-cc1" "-triple" "powerpc-unknown-netbsd"
// S-POWERPC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-POWERPC: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC64: "-cc1" "-triple" "powerpc64-unknown-netbsd"
// S-POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-POWERPC64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// Check that the driver passes include paths to cc1 on NetBSD.
// RUN: %clang -### %s --target=x86_64-unknown-netbsd -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER-PASS-INCLUDES
// DRIVER-PASS-INCLUDES:      "-cc1" {{.*}}"-resource-dir" "[[RESOURCE:[^"]+]]"
// DRIVER-PASS-INCLUDES:      "-internal-isystem" "[[RESOURCE]]{{/|\\\\}}include"
// DRIVER-PASS-INCLUDES:      "-internal-externc-isystem" "{{.*}}/usr/include"

// Test NetBSD with libstdc++ when the sysroot path ends with `/`.
// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-netbsd \
// RUN:     -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/basic_netbsd_tree/ \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBSTDCXX-SYSROOT-SLASH %s
// CHECK-BASIC-LIBSTDCXX-SYSROOT-SLASH: "-cc1"
// CHECK-BASIC-LIBSTDCXX-SYSROOT-SLASH-SAME: "-isysroot" "[[SYSROOT:[^"]+/]]"
// CHECK-BASIC-LIBSTDCXX-SYSROOT-SLASH-SAME: "-internal-isystem" "[[SYSROOT]]usr/include/g++"
