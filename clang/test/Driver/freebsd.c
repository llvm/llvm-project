// RUN: %clang -no-canonical-prefixes \
// RUN:   -target aarch64-pc-freebsd11 %s                              \
// RUN:   --sysroot=%S/Inputs/basic_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM64 %s
// CHECK-ARM64: "-cc1" "-triple" "aarch64-pc-freebsd11"
// CHECK-ARM64: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARM64: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes \
// RUN:   -target powerpc-pc-freebsd8 %s    \
// RUN:   --sysroot=%S/Inputs/basic_freebsd_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC %s
// CHECK-PPC: "-cc1" "-triple" "powerpc-pc-freebsd8"
// CHECK-PPC: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-PPC: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes \
// RUN:   -target powerpc64-pc-freebsd8 %s                              \
// RUN:   --sysroot=%S/Inputs/basic_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64 %s
// CHECK-PPC64: "-cc1" "-triple" "powerpc64-pc-freebsd8"
// CHECK-PPC64: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-PPC64: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"

// RUN: %clang -no-canonical-prefixes \
// RUN:   -target powerpc64le-unknown-freebsd13 %s \
// RUN:   --sysroot=%S/Inputs/basic_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE %s
// CHECK-PPC64LE: "-cc1" "-triple" "powerpc64le-unknown-freebsd13"
// CHECK-PPC64LE: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-PPC64LE: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"

//
// Check that -m32 properly adjusts the toolchain flags.
//
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIB32 %s
// CHECK-LIB32: "-cc1" "-triple" "i386-pc-freebsd8"
// CHECK-LIB32: ld{{.*}}" {{.*}} "-m" "elf_i386_fbsd"
//
// RUN: %clang -target x86_64-pc-freebsd8 -m32 %s 2>&1 \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -print-search-dirs 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIB32PATHS %s
// CHECK-LIB32PATHS: libraries: ={{.*:?}}/usr/lib32
//
// Check that O32 MIPS uses /usr/lib32 on a 64-bit tree.
//
// RUN: %clang -target mips-freebsd12 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -print-search-dirs 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIB32PATHS %s
//
// Check that MIPS passes the correct linker emulation.
//
// RUN: %clang -target mips-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-LD %s
// CHECK-MIPS-LD: ld{{.*}}" {{.*}} "-m" "elf32btsmip_fbsd"
// RUN: %clang -target mipsel-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL-LD %s
// CHECK-MIPSEL-LD: ld{{.*}}" {{.*}} "-m" "elf32ltsmip_fbsd"
// RUN: %clang -target mips64-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-LD %s
// CHECK-MIPS64-LD: ld{{.*}}" {{.*}} "-m" "elf64btsmip_fbsd"
// RUN: %clang -target mips64el-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-LD %s
// CHECK-MIPS64EL-LD: ld{{.*}}" {{.*}} "-m" "elf64ltsmip_fbsd"
// RUN: %clang -target mips64-freebsd -mabi=n32 %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSN32-LD %s
// CHECK-MIPSN32-LD: ld{{.*}}" {{.*}} "-m" "elf32btsmipn32_fbsd"
// RUN: %clang -target mips64el-freebsd -mabi=n32 %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSN32EL-LD %s
// CHECK-MIPSN32EL-LD: ld{{.*}}" {{.*}} "-m" "elf32ltsmipn32_fbsd"
//
// Check that RISC-V passes the correct linker emulation.
//
// RUN: %clang -target riscv32-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RV32I-LD %s
// CHECK-RV32I-LD: ld{{.*}}" {{.*}} "-m" "elf32lriscv"
// RUN: %clang -target riscv64-freebsd %s -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RV64I-LD %s
// CHECK-RV64I-LD: ld{{.*}}" {{.*}} "-m" "elf64lriscv"
//
// Check that the new linker flags are passed to FreeBSD
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS8 %s
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd9 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS9 %s
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd10.0 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS9 %s
// CHECK-LDFLAGS8-NOT: --hash-style=both
// CHECK-LDFLAGS8: --enable-new-dtags
// CHECK-LDFLAGS9: --hash-style=both
// CHECK-LDFLAGS9: --enable-new-dtags
//
// Check that we do not pass --hash-style=gnu and --hash-style=both to linker
// and provide correct path to the dynamic linker for MIPS platforms.
// Also verify that we tell the assembler to target the right ISA and ABI.
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     -target mips-unknown-freebsd10.0 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: "{{[^" ]*}}ld{{[^" ]*}}"
// CHECK-MIPS: "-dynamic-linker" "{{.*}}/libexec/ld-elf.so.1"
// CHECK-MIPS-NOT: "--hash-style={{gnu|both}}"
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     -target mipsel-unknown-freebsd10.0 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL %s
// CHECK-MIPSEL: "{{[^" ]*}}ld{{[^" ]*}}"
// CHECK-MIPSEL: "-dynamic-linker" "{{.*}}/libexec/ld-elf.so.1"
// CHECK-MIPSEL-NOT: "--hash-style={{gnu|both}}"
// RUN: %clang %s -### 2>&1 \
// RUN:     -target mips64-unknown-freebsd10.0 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64 %s
// CHECK-MIPS64: "{{[^" ]*}}ld{{[^" ]*}}"
// CHECK-MIPS64: "-dynamic-linker" "{{.*}}/libexec/ld-elf.so.1"
// CHECK-MIPS64-NOT: "--hash-style={{gnu|both}}"
// RUN: %clang %s -### 2>&1 \
// RUN:     -target mips64el-unknown-freebsd10.0 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL %s
// CHECK-MIPS64EL: "{{[^" ]*}}ld{{[^" ]*}}"
// CHECK-MIPS64EL: "-dynamic-linker" "{{.*}}/libexec/ld-elf.so.1"
// CHECK-MIPS64EL-NOT: "--hash-style={{gnu|both}}"

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -static %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC: ld{{.*}}" "--eh-frame-hdr" "-Bstatic"
// CHECK-STATIC: crt1.o
// CHECK-STATIC: crtbeginT.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -shared %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED: crti.o
// CHECK-SHARED: crtbeginS.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -pie %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PIE %s
// CHECK-PIE: pie
// CHECK-PIE: Scrt1.o
// CHECK-PIE: crtbeginS.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NORMAL %s
// CHECK-NORMAL: crt1.o
// CHECK-NORMAL: crtbegin.o

// RUN: %clang %s -### -target arm-unknown-freebsd10.0 -no-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: "-cc1"{{.*}}" "-exception-model=sjlj"
// CHECK-ARM: as{{.*}}" "-mfpu=softvfp"{{.*}}"-matpcs"
// CHECK-ARM-EABI-NOT: as{{.*}}" "-mfpu=vfp"

// RUN: %clang %s -### -target arm-gnueabi-freebsd10.0 -no-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-EABI %s
// CHECK-ARM-EABI-NOT: "-cc1"{{.*}}" "-exception-model=sjlj"
// CHECK-ARM-EABI: as{{.*}}" "-mfpu=softvfp" "-meabi=5"
// CHECK-ARM-EABI-NOT: as{{.*}}" "-mfpu=vfp"
// CHECK-ARM-EABI-NOT: as{{.*}}" "-matpcs"

// RUN: %clang %s -### -target arm-gnueabihf-freebsd10.0 -no-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-EABIHF %s
// CHECK-ARM-EABIHF-NOT: "-cc1"{{.*}}" "-exception-model=sjlj"
// CHECK-ARM-EABIHF: as{{.*}}" "-mfpu=vfp" "-meabi=5"
// CHECK-ARM-EABIHF-NOT: as{{.*}}" "-mfpu=softvfp"
// CHECK-ARM-EABIHF-NOT: as{{.*}}" "-matpcs"

// RUN: %clang -target sparc-unknown-freebsd8 %s -### -fpic -no-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SPARC-PIE %s
// CHECK-SPARC-PIE: as{{.*}}" "-KPIC

// RUN: %clang -mcpu=ultrasparc -target sparc64-unknown-freebsd8 %s -### -no-integrated-as 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SPARC-CPU %s
// CHECK-SPARC-CPU: cc1{{.*}}" "-target-cpu" "ultrasparc"
// CHECK-SPARC-CPU: as{{.*}}" "-Av9a

// Check that -G flags are passed to the linker for mips
// RUN: %clang -target mips-unknown-freebsd %s -### -G0 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-G %s
// CHECK-MIPS-G: ld{{.*}}" "-G0"

// Check CPU type for MIPS
// RUN: %clang -target mips-unknown-freebsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS-CPU %s
// RUN: %clang -target mipsel-unknown-freebsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS-CPU %s
// CHECK-MIPS-CPU: "-target-cpu" "mips2"

// Check CPU type for MIPS64
// RUN: %clang -target mips64-unknown-freebsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64-CPU %s
// RUN: %clang -target mips64el-unknown-freebsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-MIPS64-CPU %s
// CHECK-MIPS64-CPU: "-target-cpu" "mips3"

// Check that the integrated assembler is enabled for SPARC64
// RUN: %clang -target sparc64-unknown-freebsd -### -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-IAS %s
// CHECK-IAS-NOT: "-no-integrated-as"

// RUN: %clang -target ppc64-unknown-freebsd13.0 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=PPC64-MUNWIND %s
// PPC64-MUNWIND: "-funwind-tables=2"

/// -r suppresses default -l and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=aarch64-pc-freebsd11 -r \
// RUN:   --sysroot=%S/Inputs/basic_freebsd64_tree 2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE:     "-r"
// RELOCATABLE-NOT: "-l
// RELOCATABLE-NOT: crt{{[^.]+}}.o
