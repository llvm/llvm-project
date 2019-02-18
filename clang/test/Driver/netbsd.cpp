// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd6.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-6 %s
// RUN: %clangxx -no-canonical-prefixes -target arm-unknown-netbsd6.0.0-eabi \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM %s
// RUN: %clangxx -no-canonical-prefixes -target arm-unknown-netbsd7.0.0-eabi \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-7 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64_be-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64_be-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE-7 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd6.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-6 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC-7 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd6.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64-6 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd7.0.0 \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target powerpc-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC %s
// RUN: %clangxx -no-canonical-prefixes -target powerpc64-unknown-netbsd \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC64 %s

// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target x86_64-unknown-netbsd6.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-6 %s
// RUN: %clangxx -no-canonical-prefixes -target arm-unknown-netbsd6.0.0-eabi -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM %s
// RUN: %clangxx -no-canonical-prefixes -target arm-unknown-netbsd7.0.0-eabi -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-7 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64_be-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE %s
// RUN: %clangxx -no-canonical-prefixes -target aarch64_be-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE-7 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd6.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC-6 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC-7 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd6.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64-6 %s
// RUN: %clangxx -no-canonical-prefixes -target sparc64-unknown-netbsd7.0.0 -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64-7 %s
// RUN: %clangxx -no-canonical-prefixes -target powerpc-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC %s
// RUN: %clangxx -no-canonical-prefixes -target powerpc64-unknown-netbsd -static \
// RUN: -stdlib=platform --sysroot=%S/Inputs/basic_netbsd_tree %s -### 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC64 %s

// X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd6.0.0"
// X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// X86_64-6: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM: clang{{.*}}" "-cc1" "-triple" "armv5e-unknown-netbsd6.0.0-eabi"
// ARM: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// ARM: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// ARM-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++" "-lm" "-lc"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64-unknown-netbsd"
// AARCH64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64: "-lm" "-lc"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64-7: "-lm" "-lc"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE: clang{{.*}}" "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64_BE: "-lm" "-lc"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE-7: clang{{.*}}" "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// AARCH64_BE-7: "-lm" "-lc"
// AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd"
// SPARC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC: "-lm" "-lc"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC-7: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd7.0.0"
// SPARC-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC-7: "-lm" "-lc"
// SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC-6: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd6.0.0"
// SPARC-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// SPARC-6: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd"
// SPARC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC64: "-lm" "-lc"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64-7: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd7.0.0"
// SPARC64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// SPARC64-7: "-lm" "-lc"
// SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64-6: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd6.0.0"
// SPARC64-6: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// SPARC64-6: "-lm" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// SPARC64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC: clang{{.*}}" "-cc1" "-triple" "powerpc-unknown-netbsd"
// POWERPC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// POWERPC: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC64: clang{{.*}}" "-cc1" "-triple" "powerpc64-unknown-netbsd"
// POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// POWERPC64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd"
// S-X86_64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-X86_64-7: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-6: clang{{.*}}" "-cc1" "-triple" "x86_64-unknown-netbsd6.0.0"
// S-X86_64-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-X86_64-6: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-X86_64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM: clang{{.*}}" "-cc1" "-triple" "armv5e-unknown-netbsd6.0.0-eabi"
// S-ARM: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-ARM: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-7: clang{{.*}}" "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// S-ARM-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++" "-lm" "-lc"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64: clang{{.*}}" "-cc1" "-triple" "aarch64-unknown-netbsd"
// S-AARCH64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64: "-lm" "-lc"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64-7: clang{{.*}}" "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// S-AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64-7: "-lm" "-lc"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE: clang{{.*}}" "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// S-AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64_BE: "-lm" "-lc"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE-7: clang{{.*}}" "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// S-AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-AARCH64_BE-7: "-lm" "-lc"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd"
// S-SPARC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC: "-lm" "-lc"
// S-SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC-7: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd7.0.0"
// S-SPARC-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC-7: "-lm" "-lc"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC-6: clang{{.*}}" "-cc1" "-triple" "sparc-unknown-netbsd6.0.0"
// S-SPARC-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-SPARC-6: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd"
// S-SPARC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC64: "-lm" "-lc"
// S-SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64-7: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd7.0.0"
// S-SPARC64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-SPARC64-7: "-lm" "-lc"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64-6: clang{{.*}}" "-cc1" "-triple" "sparc64-unknown-netbsd6.0.0"
// S-SPARC64-6: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64-6: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64-6: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lstdc++"
// S-SPARC64-6: "-lm" "-lc" "-lgcc_eh" "-lc" "-lgcc"
// S-SPARC64-6: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC: clang{{.*}}" "-cc1" "-triple" "powerpc-unknown-netbsd"
// S-POWERPC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-POWERPC: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC64: clang{{.*}}" "-cc1" "-triple" "powerpc64-unknown-netbsd"
// S-POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc++"
// S-POWERPC64: "-lm" "-lc" "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
