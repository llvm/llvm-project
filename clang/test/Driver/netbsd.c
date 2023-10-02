// RUN: %clang --target=x86_64-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=STATIC %s
// RUN: %clang --target=x86_64-unknown-netbsd \
// RUN: -pie --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=PIE %s
// RUN: %clang --target=x86_64-unknown-netbsd \
// RUN: -static -pie --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=STATIC-PIE %s
// RUN: %clang --target=x86_64-unknown-netbsd \
// RUN: -shared --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SHARED %s

// RUN: %clang --target=x86_64-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64 %s
// RUN: %clang --target=x86_64-unknown-netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=X86_64-7 %s
// RUN: %clang --target=aarch64-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64 %s
// RUN: %clang --target=aarch64-unknown-netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64-7 %s
// RUN: %clang --target=aarch64_be-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE %s
// RUN: %clang --target=aarch64_be-unknown-netbsd7.0.0 \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=AARCH64_BE-7 %s
// RUN: %clang --target=arm-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARM %s
// RUN: %clang --target=armeb-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARMEB %s
// RUN: %clang --target=armeb-unknown-netbsd-eabi -march=armv7 \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARMV7EB %s
// RUN: %clang --target=armv7eb-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARMV7EB %s
// RUN: %clang -r --target=armeb-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARMEB-R %s
// RUN: %clang --target=arm-unknown-netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-APCS %s
// RUN: %clang --target=arm-unknown-netbsd-eabihf \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-HF %s
// RUN: %clang --target=thumb-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=THUMB %s
// RUN: %clang --target=thumbeb-unknown-netbsd-eabi \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=THUMBEB %s
// RUN: %clang --target=arm-unknown-netbsd7.0.0-eabi \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=ARM-7 %s
// RUN: %clang --target=riscv32-unknown-netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=RISCV32 %s
// RUN: %clang --target=riscv64-unknown-netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=RISCV64 %s
// RUN: %clang --target=sparc-unknown-netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC %s
// RUN: %clang --target=sparc64-unknown-netbsd \
// RUN: -no-integrated-as --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=SPARC64 %s
// RUN: %clang --target=powerpc-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC %s
// RUN: %clang --target=powerpc64-unknown-netbsd \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=POWERPC64 %s

// RUN: %clang --target=x86_64-unknown-netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64 %s
// RUN: %clang --target=x86_64-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-X86_64-7 %s
// RUN: %clang --target=aarch64-unknown-netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64 %s
// RUN: %clang --target=aarch64-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64-7 %s
// RUN: %clang --target=aarch64_be-unknown-netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE %s
// RUN: %clang --target=aarch64_be-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-AARCH64_BE-7 %s
// RUN: %clang --target=arm-unknown-netbsd-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM %s
// RUN: %clang --target=armeb-unknown-netbsd-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARMEB %s
// RUN: %clang --target=arm-unknown-netbsd7.0.0-eabi -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-ARM-7 %s
// RUN: %clang --target=riscv32-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-RISCV32-7 %s
// RUN: %clang --target=riscv64-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-RISCV64-7 %s
// RUN: %clang --target=sparc-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC-7 %s
// RUN: %clang --target=sparc64-unknown-netbsd7.0.0 -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-SPARC64-7 %s
// RUN: %clang --target=powerpc-unknown-netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC %s
// RUN: %clang --target=powerpc64-unknown-netbsd -static \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=S-POWERPC64 %s
// RUN: %clang --target=x86_64-unknown-netbsd -pthread -dM -E %s \
// RUN: | FileCheck -check-prefix=PTHREAD %s

// STATIC: ld{{.*}}" "--eh-frame-hdr"
// STATIC-NOT: "-pie"
// STATIC-NOT: "-shared"
// STATIC: "-dynamic-linker" "/libexec/ld.elf_so"
// STATIC-NOT: "-pie"
// STATIC-NOT: "-shared"
// STATIC: "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// STATIC: "{{.*}}/usr/lib{{/|\\\\}}crti.o" "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o"
// STATIC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// STATIC-PIE: ld{{.*}}" "--eh-frame-hdr"
// STATIC-PIE-NOT: "-dynamic-linker" "/libexec/ld.elf_so"
// STATIC-PIE-NOT: "-shared"
// STATIC-PIE: "-pie"
// STATIC-PIE-NOT: "-dynamic-linker" "/libexec/ld.elf_so"
// STATIC-PIE-NOT: "-shared"
// STATIC-PIE: "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// STATIC-PIE: "{{.*}}/usr/lib{{/|\\\\}}crti.o" "{{.*}}/usr/lib{{/|\\\\}}crtbeginS.o"
// STATIC-PIE: "{{.*}}/usr/lib{{/|\\\\}}crtendS.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SHARED: ld{{.*}}" "--eh-frame-hdr"
// SHARED-NOT: "-pie"
// SHARED-NOT: "-dynamic-linker"
// SHARED-NOT: "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SHARED: "{{.*}}/usr/lib{{/|\\\\}}crti.o" "{{.*}}/usr/lib{{/|\\\\}}crtbeginS.o"
// SHARED: "{{.*}}/usr/lib{{/|\\\\}}crtendS.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// PIE: ld{{.*}}" "--eh-frame-hdr"
// PIE-NOT: "-shared"
// PIE: "-pie" "-dynamic-linker" "/libexec/ld.elf_so"
// PIE-NOT: "-shared"
// PIE: "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// PIE: "{{.*}}/usr/lib{{/|\\\\}}crtbeginS.o"
// PIE: "{{.*}}/usr/lib{{/|\\\\}}crtendS.o"
// PIE: "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64: "-cc1" "-triple" "x86_64-unknown-netbsd"
// X86_64-NOT: "-fno-use-init-array"
// X86_64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// X86_64-7: "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// X86_64-7: "-fno-use-init-array"
// X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// X86_64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64: "-cc1" "-triple" "aarch64-unknown-netbsd"
// AARCH64-NOT: "-fno-use-init-array"
// AARCH64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64-7: "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// AARCH64-7-NOT: "-fno-use-init-array"
// AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE: "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// AARCH64_BE-NOT: "-fno-use-init-array"
// AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// AARCH64_BE-7: "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// AARCH64_BE-7-NOT: "-fno-use-init-array"
// AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// AARCH64_BE-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM: "-cc1" "-triple" "armv5e-unknown-netbsd-eabi"
// ARM-NOT: "-fno-use-init-array"
// ARM: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARM: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM: "-m" "armelf_nbsd_eabi"
// ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARMEB: "-cc1" "-triple" "armebv5e-unknown-netbsd-eabi"
// ARMEB-NOT: "-fno-use-init-array"
// ARMEB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARMEB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARMEB-NOT: "--be8"
// ARMEB: "-m" "armelfb_nbsd_eabi"
// ARMEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"
// ARMV7EB: as{{.*}}" "-mcpu=cortex-a8"
// ARMV7EB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARMV7EB: "--be8" "-m" "armelfb_nbsd_eabi"

// ARMEB-R: ld{{.*}}"
// ARMEB-R-NOT: "--be8"

// ARM-APCS: "-cc1" "-triple" "armv4-unknown-netbsd"
// ARM-APCS: as{{.*}}" "-mcpu=strongarm" "-o"
// ARM-APCS: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-APCS: "-m" "armelf_nbsd"
// ARM-APCS: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}oabi{{/|\\\\}}crti.o"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-APCS: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-HF: "-cc1" "-triple" "armv5e-unknown-netbsd-eabihf"
// ARM-HF: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// ARM-HF: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-HF: "-m" "armelf_nbsd_eabihf"
// ARM-HF: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}eabihf{{/|\\\\}}crti.o"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-HF: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// THUMB: "-cc1" "-triple" "armv5e-unknown-netbsd-eabi"
// THUMB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// THUMB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// THUMB: "-m" "armelf_nbsd_eabi"
// THUMB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// THUMB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// THUMBEB: "-cc1" "-triple" "armebv5e-unknown-netbsd-eabi"
// THUMBEB: as{{.*}}" "-mcpu=arm926ej-s" "-o"
// THUMBEB: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// THUMBEB: "-m" "armelfb_nbsd_eabi"
// THUMBEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// THUMBEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// ARM-7: "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// ARM-7-NOT: "-fno-use-init-array"
// ARM-7: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// ARM-7: "-m" "armelf_nbsd_eabi"
// ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// ARM-7:  "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// RISCV32: "-cc1" "-triple" "riscv32-unknown-netbsd"
// RISCV32-NOT: "-fno-use-init-array"
// RISCV32: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// RISCV32-SAME: "-m" "elf32lriscv" "-X"
// RISCV32-SAME: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// RISCV32-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// RISCV32-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// RISCV64: "-cc1" "-triple" "riscv64-unknown-netbsd"
// RISCV64-NOT: "-fno-use-init-array"
// RISCV64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// RISCV64-SAME: "-m" "elf64lriscv" "-X"
// RISCV64-SAME: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// RISCV64-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// RISCV64-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC: "-cc1" "-triple" "sparc-unknown-netbsd"
// SPARC-NOT: "-fno-use-init-array"
// SPARC: as{{.*}}" "-32" "-Av8" "-o"
// SPARC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC: "-m" "elf32_sparc"
// SPARC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// SPARC64: "-cc1" "-triple" "sparc64-unknown-netbsd"
// SPARC64-NOT: "-fno-use-init-array"
// SPARC64: as{{.*}}" "-64" "-Av9" "-o"
// SPARC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// SPARC64: "-m" "elf64_sparc"
// SPARC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// SPARC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC: "-cc1" "-triple" "powerpc-unknown-netbsd"
// POWERPC-NOT: "-fno-use-init-array"
// POWERPC: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC: "-m" "elf32ppc_nbsd"
// POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// POWERPC64: "-cc1" "-triple" "powerpc64-unknown-netbsd"
// POWERPC64-NOT: "-fno-use-init-array"
// POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-dynamic-linker" "/libexec/ld.elf_so"
// POWERPC64: "-m" "elf64ppc"
// POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64: "-cc1" "-triple" "x86_64-unknown-netbsd"
// S-X86_64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-X86_64-7: "-cc1" "-triple" "x86_64-unknown-netbsd7.0.0"
// S-X86_64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-X86_64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-X86_64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64: "-cc1" "-triple" "aarch64-unknown-netbsd"
// S-AARCH64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64-7: "-cc1" "-triple" "aarch64-unknown-netbsd7.0.0"
// S-AARCH64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE: "-cc1" "-triple" "aarch64_be-unknown-netbsd"
// S-AARCH64_BE: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64_BE: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-AARCH64_BE-7: "-cc1" "-triple" "aarch64_be-unknown-netbsd7.0.0"
// S-AARCH64_BE-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-AARCH64_BE-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-AARCH64_BE-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM: "-cc1" "-triple" "armv5e-unknown-netbsd-eabi"
// S-ARM: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM: "-m" "armelf_nbsd_eabi"
// S-ARM: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARMEB: "-cc1" "-triple" "armebv5e-unknown-netbsd-eabi"
// S-ARMEB: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARMEB: "-m" "armelfb_nbsd_eabi"
// S-ARMEB: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARMEB: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-ARM-7: "-cc1" "-triple" "armv5e-unknown-netbsd7.0.0-eabi"
// S-ARM-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-ARM-7: "-m" "armelf_nbsd_eabi"
// S-ARM-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}eabi{{/|\\\\}}crti.o"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-ARM-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-RISCV32-7: "-cc1" "-triple" "riscv32-unknown-netbsd7.0.0"
// S-RISCV32-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-RISCV32-7-SAME: "-m" "elf32lriscv" "-X"
// S-RISCV32-7-SAME: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-RISCV32-7-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-RISCV32-7-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-RISCV64-7: "-cc1" "-triple" "riscv64-unknown-netbsd7.0.0"
// S-RISCV64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-RISCV64-7-SAME: "-m" "elf64lriscv" "-X"
// S-RISCV64-7-SAME: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-RISCV64-7-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-RISCV64-7-SAME: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC-7: "-cc1" "-triple" "sparc-unknown-netbsd7.0.0"
// S-SPARC-7: "-fno-use-init-array"
// S-SPARC-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC-7: "-m" "elf32_sparc"
// S-SPARC-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}sparc{{/|\\\\}}crti.o"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-SPARC64-7: "-cc1" "-triple" "sparc64-unknown-netbsd7.0.0"
// S-SPARC64-7: "-fno-use-init-array"
// S-SPARC64-7: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-SPARC64-7: "-m" "elf64_sparc"
// S-SPARC64-7: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o" "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-SPARC64-7: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC: "-cc1" "-triple" "powerpc-unknown-netbsd"
// S-POWERPC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC: "-m" "elf32ppc_nbsd"
// S-POWERPC: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}powerpc{{/|\\\\}}crti.o"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-POWERPC: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// S-POWERPC64: "-cc1" "-triple" "powerpc64-unknown-netbsd"
// S-POWERPC64: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// S-POWERPC64: "-m" "elf64ppc"
// S-POWERPC64: "-o" "a.out" "{{.*}}/usr/lib{{/|\\\\}}crt0.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtbegin.o" "{{.*}}.o" "-lc"
// S-POWERPC64: "{{.*}}/usr/lib{{/|\\\\}}crtend.o" "{{.*}}/usr/lib{{/|\\\\}}crtn.o"

// PTHREAD-NOT: _POSIX_THREADS
// PTHREAD:     _REENTRANT
// PTHREAD-NOT: _POSIX_THREADS

// Check PowerPC for Secure PLT
// RUN: %clang --target=powerpc-unknown-netbsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=POWERPC-SECUREPLT %s
// POWERPC-SECUREPLT: "-target-feature" "+secure-plt"

// -r suppresses -dynamic-linker, default -l and crt*.o like -nostdlib.
// RUN: %clang --target=x86_64-unknown-netbsd -r \
// RUN: --sysroot=%S/Inputs/basic_netbsd_tree -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=RELOCATABLE %s
// RELOCATABLE:     "-r"
// RELOCATABLE-NOT: "-pie"
// RELOCATABLE-NOT: "-dynamic-linker"
// RELOCATABLE-NOT: "-l
// RELOCATABLE-NOT: crt{{[^./\\]+}}.o

// Check that the driver passes include paths to cc1 on NetBSD.
// RUN: %clang -### %s --target=x86_64-unknown-netbsd -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER-PASS-INCLUDES
// DRIVER-PASS-INCLUDES:      "-cc1" {{.*}}"-resource-dir" "[[RESOURCE:[^"]+]]"
// DRIVER-PASS-INCLUDES-SAME: "-internal-isystem" "[[RESOURCE]]{{/|\\\\}}include"
// DRIVER-PASS-INCLUDES-SAME: {{^}} "-internal-externc-isystem" "{{.*}}/usr/include"
