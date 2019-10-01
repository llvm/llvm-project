// Check frontend and linker invocations on the MTI MIPS toolchain.

// -EB -mhard-float -mabi=32
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -mabi=32 \
// RUN:   | FileCheck --check-prefix=EB-HARD-O32 %s
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard/lib"
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-O32: "-internal-externc-isystem"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r2-hard/lib/../usr/include"
// EB-HARD-O32: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-O32: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard"
// EB-HARD-O32: "-dynamic-linker" "/lib/ld.so.1"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-O32: "[[TC]]/mips-r2-hard/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-O32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard/lib"
// EB-HARD-O32: "-L[[TC]]/mips-r2-hard/lib"
// EB-HARD-O32: "-L[[TC]]/../../../../sysroot/mips-r2-hard/lib/../lib"
// EB-HARD-O32: "-L[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib"
// EB-HARD-O32: "[[TC]]/mips-r2-hard/lib{{/|\\\\}}crtend.o"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mhard-float -mabi=n32
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -mabi=n32 \
// RUN:   | FileCheck --check-prefix=EB-HARD-N32 %s
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard/lib32"
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-N32: "-internal-externc-isystem"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r2-hard/lib32/../usr/include"
// EB-HARD-N32: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-N32: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard"
// EB-HARD-N32: "-dynamic-linker" "/lib32/ld.so.1"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib32{{/|\\\\}}crt1.o"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib32{{/|\\\\}}crti.o"
// EB-HARD-N32: "[[TC]]/mips-r2-hard/lib32{{/|\\\\}}crtbegin.o"
// EB-HARD-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard/lib32"
// EB-HARD-N32: "-L[[TC]]/mips-r2-hard/lib32"
// EB-HARD-N32: "-L[[TC]]/../../../../sysroot/mips-r2-hard/lib/../lib32"
// EB-HARD-N32: "-L[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib32"
// EB-HARD-N32: "[[TC]]/mips-r2-hard/lib32{{/|\\\\}}crtend.o"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib32{{/|\\\\}}crtn.o"

// -EB -mhard-float -mabi=64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips64-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -mabi=64 \
// RUN:   | FileCheck --check-prefix=EB-HARD-N64 %s
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard/lib64"
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-N64: "-internal-externc-isystem"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r2-hard/lib64/../usr/include"
// EB-HARD-N64: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-N64: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard"
// EB-HARD-N64: "-dynamic-linker" "/lib64/ld.so.1"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib64{{/|\\\\}}crt1.o"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib64{{/|\\\\}}crti.o"
// EB-HARD-N64: "[[TC]]/mips-r2-hard/lib64{{/|\\\\}}crtbegin.o"
// EB-HARD-N64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard/lib64"
// EB-HARD-N64: "-L[[TC]]/mips-r2-hard/lib64"
// EB-HARD-N64: "-L[[TC]]/../../../../sysroot/mips-r2-hard/lib/../lib64"
// EB-HARD-N64: "-L[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib64"
// EB-HARD-N64: "[[TC]]/mips-r2-hard/lib64{{/|\\\\}}crtend.o"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r2-hard/usr/lib/../lib64{{/|\\\\}}crtn.o"

// -EL -mhard-float -mabi=32
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -mabi=32 \
// RUN:   | FileCheck --check-prefix=EL-HARD-O32 %s
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard/lib"
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-O32: "-internal-externc-isystem"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/lib/../usr/include"
// EL-HARD-O32: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-O32: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard"
// EL-HARD-O32: "-dynamic-linker" "/lib/ld.so.1"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-O32: "[[TC]]/mipsel-r2-hard/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-O32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard/lib"
// EL-HARD-O32: "-L[[TC]]/mipsel-r2-hard/lib"
// EL-HARD-O32: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/lib/../lib"
// EL-HARD-O32: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib"
// EL-HARD-O32: "[[TC]]/mipsel-r2-hard/lib{{/|\\\\}}crtend.o"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mhard-float -mabi=n32
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -mabi=n32 \
// RUN:   | FileCheck --check-prefix=EL-HARD-N32 %s
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard/lib32"
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-N32: "-internal-externc-isystem"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/lib32/../usr/include"
// EL-HARD-N32: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-N32: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard"
// EL-HARD-N32: "-dynamic-linker" "/lib32/ld.so.1"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib32{{/|\\\\}}crt1.o"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib32{{/|\\\\}}crti.o"
// EL-HARD-N32: "[[TC]]/mipsel-r2-hard/lib32{{/|\\\\}}crtbegin.o"
// EL-HARD-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard/lib32"
// EL-HARD-N32: "-L[[TC]]/mipsel-r2-hard/lib32"
// EL-HARD-N32: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/lib/../lib32"
// EL-HARD-N32: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib32"
// EL-HARD-N32: "[[TC]]/mipsel-r2-hard/lib32{{/|\\\\}}crtend.o"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib32{{/|\\\\}}crtn.o"

// -EL -mhard-float -mabi=64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips64-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -mabi=64 \
// RUN:   | FileCheck --check-prefix=EL-HARD-N64 %s
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard/lib64"
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-N64: "-internal-externc-isystem"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r2-hard/lib64/../usr/include"
// EL-HARD-N64: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-N64: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard"
// EL-HARD-N64: "-dynamic-linker" "/lib64/ld.so.1"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib64{{/|\\\\}}crt1.o"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib64{{/|\\\\}}crti.o"
// EL-HARD-N64: "[[TC]]/mipsel-r2-hard/lib64{{/|\\\\}}crtbegin.o"
// EL-HARD-N64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard/lib64"
// EL-HARD-N64: "-L[[TC]]/mipsel-r2-hard/lib64"
// EL-HARD-N64: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/lib/../lib64"
// EL-HARD-N64: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib64"
// EL-HARD-N64: "[[TC]]/mipsel-r2-hard/lib64{{/|\\\\}}crtend.o"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r2-hard/usr/lib/../lib64{{/|\\\\}}crtn.o"

// -EB -msoft-float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -msoft-float \
// RUN:   | FileCheck --check-prefix=EB-SOFT %s
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-soft/lib"
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-SOFT: "-internal-externc-isystem"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r2-soft/lib/../usr/include"
// EB-SOFT: "{{.*}}ld{{(.exe)?}}"
// EB-SOFT: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-soft"
// EB-SOFT: "-dynamic-linker" "/lib/ld.so.1"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r2-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r2-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-SOFT: "[[TC]]/mips-r2-soft/lib{{/|\\\\}}crtbegin.o"
// EB-SOFT: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-soft/lib"
// EB-SOFT: "-L[[TC]]/mips-r2-soft/lib"
// EB-SOFT: "-L[[TC]]/../../../../sysroot/mips-r2-soft/lib/../lib"
// EB-SOFT: "-L[[TC]]/../../../../sysroot/mips-r2-soft/usr/lib/../lib"
// EB-SOFT: "[[TC]]/mips-r2-soft/lib{{/|\\\\}}crtend.o"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r2-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -msoft-float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -msoft-float \
// RUN:   | FileCheck --check-prefix=EL-SOFT %s
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-soft/lib"
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-SOFT: "-internal-externc-isystem"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r2-soft/lib/../usr/include"
// EL-SOFT: "{{.*}}ld{{(.exe)?}}"
// EL-SOFT: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-soft"
// EL-SOFT: "-dynamic-linker" "/lib/ld.so.1"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-SOFT: "[[TC]]/mipsel-r2-soft/lib{{/|\\\\}}crtbegin.o"
// EL-SOFT: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-soft/lib"
// EL-SOFT: "-L[[TC]]/mipsel-r2-soft/lib"
// EL-SOFT: "-L[[TC]]/../../../../sysroot/mipsel-r2-soft/lib/../lib"
// EL-SOFT: "-L[[TC]]/../../../../sysroot/mipsel-r2-soft/usr/lib/../lib"
// EL-SOFT: "[[TC]]/mipsel-r2-soft/lib{{/|\\\\}}crtend.o"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mhard-float -muclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -muclibc \
// RUN:   | FileCheck --check-prefix=EB-HARD-UCLIBC %s
// EB-HARD-UCLIBC: "-internal-isystem"
// EB-HARD-UCLIBC: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-UCLIBC: "-internal-isystem"
// EB-HARD-UCLIBC: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard-uclibc/lib"
// EB-HARD-UCLIBC: "-internal-isystem"
// EB-HARD-UCLIBC: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-UCLIBC: "-internal-externc-isystem"
// EB-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/lib/../usr/include"
// EB-HARD-UCLIBC: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-UCLIBC: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard-uclibc"
// EB-HARD-UCLIBC: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// EB-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-UCLIBC: "[[TC]]/mips-r2-hard-uclibc/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-UCLIBC: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard-uclibc/lib"
// EB-HARD-UCLIBC: "-L[[TC]]/mips-r2-hard-uclibc/lib"
// EB-HARD-UCLIBC: "-L[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/lib/../lib"
// EB-HARD-UCLIBC: "-L[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/usr/lib/../lib"
// EB-HARD-UCLIBC: "[[TC]]/mips-r2-hard-uclibc/lib{{/|\\\\}}crtend.o"
// EB-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mips-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mhard-float -muclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -muclibc \
// RUN:   | FileCheck --check-prefix=EL-HARD-UCLIBC %s
// EL-HARD-UCLIBC: "-internal-isystem"
// EL-HARD-UCLIBC: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-UCLIBC: "-internal-isystem"
// EL-HARD-UCLIBC: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard-uclibc/lib"
// EL-HARD-UCLIBC: "-internal-isystem"
// EL-HARD-UCLIBC: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-UCLIBC: "-internal-externc-isystem"
// EL-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/lib/../usr/include"
// EL-HARD-UCLIBC: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-UCLIBC: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc"
// EL-HARD-UCLIBC: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// EL-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-UCLIBC: "[[TC]]/mipsel-r2-hard-uclibc/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-UCLIBC: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard-uclibc/lib"
// EL-HARD-UCLIBC: "-L[[TC]]/mipsel-r2-hard-uclibc/lib"
// EL-HARD-UCLIBC: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/lib/../lib"
// EL-HARD-UCLIBC: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/usr/lib/../lib"
// EL-HARD-UCLIBC: "[[TC]]/mipsel-r2-hard-uclibc/lib{{/|\\\\}}crtend.o"
// EL-HARD-UCLIBC: "[[TC]]/../../../../sysroot/mipsel-r2-hard-uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mhard-float -mnan=2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -mnan=2008 \
// RUN:   | FileCheck --check-prefix=EB-HARD-NAN2008 %s
// EB-HARD-NAN2008: "-internal-isystem"
// EB-HARD-NAN2008: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-NAN2008: "-internal-isystem"
// EB-HARD-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard-nan2008/lib"
// EB-HARD-NAN2008: "-internal-isystem"
// EB-HARD-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-NAN2008: "-internal-externc-isystem"
// EB-HARD-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/lib/../usr/include"
// EB-HARD-NAN2008: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-NAN2008: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard-nan2008"
// EB-HARD-NAN2008: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EB-HARD-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-NAN2008: "[[TC]]/mips-r2-hard-nan2008/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-NAN2008: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard-nan2008/lib"
// EB-HARD-NAN2008: "-L[[TC]]/mips-r2-hard-nan2008/lib"
// EB-HARD-NAN2008: "-L[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/lib/../lib"
// EB-HARD-NAN2008: "-L[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/usr/lib/../lib"
// EB-HARD-NAN2008: "[[TC]]/mips-r2-hard-nan2008/lib{{/|\\\\}}crtend.o"
// EB-HARD-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mhard-float -mnan=2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -mnan=2008 \
// RUN:   | FileCheck --check-prefix=EL-HARD-NAN2008 %s
// EL-HARD-NAN2008: "-internal-isystem"
// EL-HARD-NAN2008: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-NAN2008: "-internal-isystem"
// EL-HARD-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard-nan2008/lib"
// EL-HARD-NAN2008: "-internal-isystem"
// EL-HARD-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-NAN2008: "-internal-externc-isystem"
// EL-HARD-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/lib/../usr/include"
// EL-HARD-NAN2008: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-NAN2008: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008"
// EL-HARD-NAN2008: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-HARD-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-NAN2008: "[[TC]]/mipsel-r2-hard-nan2008/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-NAN2008: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard-nan2008/lib"
// EL-HARD-NAN2008: "-L[[TC]]/mipsel-r2-hard-nan2008/lib"
// EL-HARD-NAN2008: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/lib/../lib"
// EL-HARD-NAN2008: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/usr/lib/../lib"
// EL-HARD-NAN2008: "[[TC]]/mipsel-r2-hard-nan2008/lib{{/|\\\\}}crtend.o"
// EL-HARD-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mhard-float -muclibc -mnan=2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mhard-float -muclibc -mnan=2008 \
// RUN:   | FileCheck --check-prefix=EB-HARD-UCLIBC-NAN2008 %s
// EB-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EB-HARD-UCLIBC-NAN2008: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EB-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mips-r2-hard-nan2008-uclibc/lib"
// EB-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-UCLIBC-NAN2008: "-internal-externc-isystem"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/lib/../usr/include"
// EB-HARD-UCLIBC-NAN2008: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-UCLIBC-NAN2008: "--sysroot=[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc"
// EB-HARD-UCLIBC-NAN2008: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/mips-r2-hard-nan2008-uclibc/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips-r2-hard-nan2008-uclibc/lib"
// EB-HARD-UCLIBC-NAN2008: "-L[[TC]]/mips-r2-hard-nan2008-uclibc/lib"
// EB-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/lib/../lib"
// EB-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/usr/lib/../lib"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/mips-r2-hard-nan2008-uclibc/lib{{/|\\\\}}crtend.o"
// EB-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mips-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mhard-float -muclibc -mnan=2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -muclibc -mnan=2008 \
// RUN:   | FileCheck --check-prefix=EL-HARD-UCLIBC-NAN2008 %s
// EL-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EL-HARD-UCLIBC-NAN2008: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/mipsel-r2-hard-nan2008-uclibc/lib"
// EL-HARD-UCLIBC-NAN2008: "-internal-isystem"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-UCLIBC-NAN2008: "-internal-externc-isystem"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/lib/../usr/include"
// EL-HARD-UCLIBC-NAN2008: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-UCLIBC-NAN2008: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc"
// EL-HARD-UCLIBC-NAN2008: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/mipsel-r2-hard-nan2008-uclibc/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mipsel-r2-hard-nan2008-uclibc/lib"
// EL-HARD-UCLIBC-NAN2008: "-L[[TC]]/mipsel-r2-hard-nan2008-uclibc/lib"
// EL-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/lib/../lib"
// EL-HARD-UCLIBC-NAN2008: "-L[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/usr/lib/../lib"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/mipsel-r2-hard-nan2008-uclibc/lib{{/|\\\\}}crtend.o"
// EL-HARD-UCLIBC-NAN2008: "[[TC]]/../../../../sysroot/mipsel-r2-hard-nan2008-uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -msoft-float -mmicromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -msoft-float -mmicromips \
// RUN:   | FileCheck --check-prefix=EL-SOFT-MICRO %s
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/micromipsel-r2-soft/lib"
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-SOFT-MICRO: "-internal-externc-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r2-soft/lib/../usr/include"
// EL-SOFT-MICRO: "{{.*}}ld{{(.exe)?}}"
// EL-SOFT-MICRO: "--sysroot=[[TC]]/../../../../sysroot/micromipsel-r2-soft"
// EL-SOFT-MICRO: "-dynamic-linker" "/lib/ld.so.1"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-SOFT-MICRO: "[[TC]]/micromipsel-r2-soft/lib{{/|\\\\}}crtbegin.o"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/micromipsel-r2-soft/lib"
// EL-SOFT-MICRO: "-L[[TC]]/micromipsel-r2-soft/lib"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r2-soft/lib/../lib"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r2-soft/usr/lib/../lib"
// EL-SOFT-MICRO: "[[TC]]/micromipsel-r2-soft/lib{{/|\\\\}}crtend.o"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r2-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mhard-float -mmicromips -mnan=2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:        --target=mips-mti-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_mti_tree \
// RUN:        --sysroot="" \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mhard-float -mmicromips -mnan=2008 \
// RUN:   | FileCheck --check-prefix=EL-SOFT-MICRO-NAN2008 %s
// EL-SOFT-MICRO-NAN2008: "-internal-isystem"
// EL-SOFT-MICRO-NAN2008: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.2]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2"
// EL-SOFT-MICRO-NAN2008: "-internal-isystem"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/mips-mti-linux-gnu/micromipsel-r2-hard-nan2008/lib"
// EL-SOFT-MICRO-NAN2008: "-internal-isystem"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.2/backward"
// EL-SOFT-MICRO-NAN2008: "-internal-externc-isystem"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/lib/../usr/include"
// EL-SOFT-MICRO-NAN2008: "{{.*}}ld{{(.exe)?}}"
// EL-SOFT-MICRO-NAN2008: "--sysroot=[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008"
// EL-SOFT-MICRO-NAN2008: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/micromipsel-r2-hard-nan2008/lib{{/|\\\\}}crtbegin.o"
// EL-SOFT-MICRO-NAN2008: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/micromipsel-r2-hard-nan2008/lib"
// EL-SOFT-MICRO-NAN2008: "-L[[TC]]/micromipsel-r2-hard-nan2008/lib"
// EL-SOFT-MICRO-NAN2008: "-L[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/lib/../lib"
// EL-SOFT-MICRO-NAN2008: "-L[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/usr/lib/../lib"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/micromipsel-r2-hard-nan2008/lib{{/|\\\\}}crtend.o"
// EL-SOFT-MICRO-NAN2008: "[[TC]]/../../../../sysroot/micromipsel-r2-hard-nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
