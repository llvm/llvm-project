// UNSUPPORTED: system-windows

// Verify that mips*-*-none-elf triples are handled by the BareMetal toolchain,
// i.e. clang drives ld.lld directly with the correct ELF emulation instead of
// delegating to a $target-gcc.

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld --target=mips-none-elf --sysroot= 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-BAREMETAL %s
// MIPS-BAREMETAL: "-cc1" "-triple" "mips-unknown-none-elf"
// MIPS-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// MIPS-BAREMETAL-SAME: "-Bstatic" "-m" "elf32btsmip"

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld --target=mipsel-none-elf --sysroot= 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPSEL-BAREMETAL %s
// MIPSEL-BAREMETAL: "-cc1" "-triple" "mipsel-unknown-none-elf"
// MIPSEL-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// MIPSEL-BAREMETAL-SAME: "-Bstatic" "-m" "elf32ltsmip"

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld --target=mips64-none-elf --sysroot= 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-BAREMETAL %s
// MIPS64-BAREMETAL: "-cc1" "-triple" "mips64-unknown-none-elf"
// MIPS64-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// MIPS64-BAREMETAL-SAME: "-Bstatic" "-m" "elf64btsmip"

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld --target=mips64el-none-elf --sysroot= 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64EL-BAREMETAL %s
// MIPS64EL-BAREMETAL: "-cc1" "-triple" "mips64el-unknown-none-elf"
// MIPS64EL-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// MIPS64EL-BAREMETAL-SAME: "-Bstatic" "-m" "elf64ltsmip"

// RUN: %clang -### %s -fuse-ld=lld -B%S/Inputs/lld --target=mips64-none-elf -mabi=n32 --sysroot= 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS64-N32-BAREMETAL %s
// MIPS64-N32-BAREMETAL: "-cc1" "-triple" "mips64-unknown-none-elf"
// MIPS64-N32-BAREMETAL: "{{.*}}/Inputs/lld/ld.lld"
// MIPS64-N32-BAREMETAL-SAME: "-Bstatic" "-m" "elf32btsmipn32"
