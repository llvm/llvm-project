// Check frontend and linker invocations on the IMG v2 MIPS toolchain.

// -EB -mips32r6 -mhard-float -mabi=32
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips32r6 -mhard-float -mabi=32 -no-pie \
// RUN:   | FileCheck --check-prefix=EB-HARD-O32 %s
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mips-r6-hard/lib"
// EB-HARD-O32: "-internal-isystem"
// EB-HARD-O32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-O32: "-internal-externc-isystem"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r6-hard/lib/../usr/include"
// EB-HARD-O32: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-O32: "--sysroot=[[TC]]/../../../../sysroot/mips-r6-hard"
// EB-HARD-O32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-O32: "[[TC]]/mips-r6-hard/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-O32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips-r6-hard/lib"
// EB-HARD-O32: "-L[[TC]]/mips-r6-hard/lib"
// EB-HARD-O32: "-L[[TC]]/../../../../sysroot/mips-r6-hard/lib/../lib"
// EB-HARD-O32: "-L[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib"
// EB-HARD-O32: "[[TC]]/mips-r6-hard/lib{{/|\\\\}}crtend.o"
// EB-HARD-O32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mips64r6 -mhard-float -mabi=n32
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips64r6 -mhard-float -mabi=n32 -no-pie \
// RUN:   | FileCheck --check-prefix=EB-HARD-N32 %s
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mips-r6-hard/lib32"
// EB-HARD-N32: "-internal-isystem"
// EB-HARD-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-N32: "-internal-externc-isystem"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r6-hard/lib32/../usr/include"
// EB-HARD-N32: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-N32: "--sysroot=[[TC]]/../../../../sysroot/mips-r6-hard"
// EB-HARD-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib32{{/|\\\\}}crt1.o"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib32{{/|\\\\}}crti.o"
// EB-HARD-N32: "[[TC]]/mips-r6-hard/lib32{{/|\\\\}}crtbegin.o"
// EB-HARD-N32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips-r6-hard/lib32"
// EB-HARD-N32: "-L[[TC]]/mips-r6-hard/lib32"
// EB-HARD-N32: "-L[[TC]]/../../../../sysroot/mips-r6-hard/lib/../lib32"
// EB-HARD-N32: "-L[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib32"
// EB-HARD-N32: "[[TC]]/mips-r6-hard/lib32{{/|\\\\}}crtend.o"
// EB-HARD-N32: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib32{{/|\\\\}}crtn.o"

// -EB -mips64r6 -mhard-float -mabi=64
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips64-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips64r6 -mhard-float -mabi=64 -no-pie \
// RUN:   | FileCheck --check-prefix=EB-HARD-N64 %s
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mips-r6-hard/lib64"
// EB-HARD-N64: "-internal-isystem"
// EB-HARD-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-N64: "-internal-externc-isystem"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r6-hard/lib64/../usr/include"
// EB-HARD-N64: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-N64: "--sysroot=[[TC]]/../../../../sysroot/mips-r6-hard"
// EB-HARD-N64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib64{{/|\\\\}}crt1.o"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib64{{/|\\\\}}crti.o"
// EB-HARD-N64: "[[TC]]/mips-r6-hard/lib64{{/|\\\\}}crtbegin.o"
// EB-HARD-N64: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips-r6-hard/lib64"
// EB-HARD-N64: "-L[[TC]]/mips-r6-hard/lib64"
// EB-HARD-N64: "-L[[TC]]/../../../../sysroot/mips-r6-hard/lib/../lib64"
// EB-HARD-N64: "-L[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib64"
// EB-HARD-N64: "[[TC]]/mips-r6-hard/lib64{{/|\\\\}}crtend.o"
// EB-HARD-N64: "[[TC]]/../../../../sysroot/mips-r6-hard/usr/lib/../lib64{{/|\\\\}}crtn.o"

// -EL -mips32r6 -mhard-float -mabi=32
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips32r6 -mhard-float -mabi=32 -no-pie \
// RUN:   | FileCheck --check-prefix=EL-HARD-O32 %s
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mipsel-r6-hard/lib"
// EL-HARD-O32: "-internal-isystem"
// EL-HARD-O32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-O32: "-internal-externc-isystem"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/lib/../usr/include"
// EL-HARD-O32: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-O32: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r6-hard"
// EL-HARD-O32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-O32: "[[TC]]/mipsel-r6-hard/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-O32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mipsel-r6-hard/lib"
// EL-HARD-O32: "-L[[TC]]/mipsel-r6-hard/lib"
// EL-HARD-O32: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/lib/../lib"
// EL-HARD-O32: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib"
// EL-HARD-O32: "[[TC]]/mipsel-r6-hard/lib{{/|\\\\}}crtend.o"
// EL-HARD-O32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mips64r6 -mhard-float -mabi=n32
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips64r6 -mhard-float -mabi=n32 -no-pie \
// RUN:   | FileCheck --check-prefix=EL-HARD-N32 %s
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mipsel-r6-hard/lib32"
// EL-HARD-N32: "-internal-isystem"
// EL-HARD-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-N32: "-internal-externc-isystem"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/lib32/../usr/include"
// EL-HARD-N32: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-N32: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r6-hard"
// EL-HARD-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib32{{/|\\\\}}crt1.o"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib32{{/|\\\\}}crti.o"
// EL-HARD-N32: "[[TC]]/mipsel-r6-hard/lib32{{/|\\\\}}crtbegin.o"
// EL-HARD-N32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mipsel-r6-hard/lib32"
// EL-HARD-N32: "-L[[TC]]/mipsel-r6-hard/lib32"
// EL-HARD-N32: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/lib/../lib32"
// EL-HARD-N32: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib32"
// EL-HARD-N32: "[[TC]]/mipsel-r6-hard/lib32{{/|\\\\}}crtend.o"
// EL-HARD-N32: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib32{{/|\\\\}}crtn.o"

// -EL -mips64r6 -mhard-float -mabi=64
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips64-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips64r6 -mhard-float -mabi=64 -no-pie \
// RUN:   | FileCheck --check-prefix=EL-HARD-N64 %s
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mipsel-r6-hard/lib64"
// EL-HARD-N64: "-internal-isystem"
// EL-HARD-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-N64: "-internal-externc-isystem"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r6-hard/lib64/../usr/include"
// EL-HARD-N64: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-N64: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r6-hard"
// EL-HARD-N64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib64{{/|\\\\}}crt1.o"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib64{{/|\\\\}}crti.o"
// EL-HARD-N64: "[[TC]]/mipsel-r6-hard/lib64{{/|\\\\}}crtbegin.o"
// EL-HARD-N64: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mipsel-r6-hard/lib64"
// EL-HARD-N64: "-L[[TC]]/mipsel-r6-hard/lib64"
// EL-HARD-N64: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/lib/../lib64"
// EL-HARD-N64: "-L[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib64"
// EL-HARD-N64: "[[TC]]/mipsel-r6-hard/lib64{{/|\\\\}}crtend.o"
// EL-HARD-N64: "[[TC]]/../../../../sysroot/mipsel-r6-hard/usr/lib/../lib64{{/|\\\\}}crtn.o"

// -EB -mips32r6 -msoft-float
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips32r6 -msoft-float -no-pie \
// RUN:   | FileCheck --check-prefix=EB-SOFT %s
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mips-r6-soft/lib"
// EB-SOFT: "-internal-isystem"
// EB-SOFT: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-SOFT: "-internal-externc-isystem"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r6-soft/lib/../usr/include"
// EB-SOFT: "{{.*}}ld{{(.exe)?}}"
// EB-SOFT: "--sysroot=[[TC]]/../../../../sysroot/mips-r6-soft"
// EB-SOFT: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r6-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r6-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-SOFT: "[[TC]]/mips-r6-soft/lib{{/|\\\\}}crtbegin.o"
// EB-SOFT: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips-r6-soft/lib"
// EB-SOFT: "-L[[TC]]/mips-r6-soft/lib"
// EB-SOFT: "-L[[TC]]/../../../../sysroot/mips-r6-soft/lib/../lib"
// EB-SOFT: "-L[[TC]]/../../../../sysroot/mips-r6-soft/usr/lib/../lib"
// EB-SOFT: "[[TC]]/mips-r6-soft/lib{{/|\\\\}}crtend.o"
// EB-SOFT: "[[TC]]/../../../../sysroot/mips-r6-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mips32r6 -msoft-float
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips32r6 -msoft-float -no-pie \
// RUN:   | FileCheck --check-prefix=EL-SOFT %s
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/mipsel-r6-soft/lib"
// EL-SOFT: "-internal-isystem"
// EL-SOFT: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-SOFT: "-internal-externc-isystem"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r6-soft/lib/../usr/include"
// EL-SOFT: "{{.*}}ld{{(.exe)?}}"
// EL-SOFT: "--sysroot=[[TC]]/../../../../sysroot/mipsel-r6-soft"
// EL-SOFT: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-SOFT: "[[TC]]/mipsel-r6-soft/lib{{/|\\\\}}crtbegin.o"
// EL-SOFT: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mipsel-r6-soft/lib"
// EL-SOFT: "-L[[TC]]/mipsel-r6-soft/lib"
// EL-SOFT: "-L[[TC]]/../../../../sysroot/mipsel-r6-soft/lib/../lib"
// EL-SOFT: "-L[[TC]]/../../../../sysroot/mipsel-r6-soft/usr/lib/../lib"
// EL-SOFT: "[[TC]]/mipsel-r6-soft/lib{{/|\\\\}}crtend.o"
// EL-SOFT: "[[TC]]/../../../../sysroot/mipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mips32r6 -mhard-float -mmicromips
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips32r6 -mhard-float -mmicromips -no-pie \
// RUN:   | FileCheck --check-prefix=EB-HARD-MICRO %s
// EB-HARD-MICRO: "-internal-isystem"
// EB-HARD-MICRO: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-HARD-MICRO: "-internal-isystem"
// EB-HARD-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/micromips-r6-hard/lib"
// EB-HARD-MICRO: "-internal-isystem"
// EB-HARD-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-HARD-MICRO: "-internal-externc-isystem"
// EB-HARD-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-hard/lib/../usr/include"
// EB-HARD-MICRO: "{{.*}}ld{{(.exe)?}}"
// EB-HARD-MICRO: "--sysroot=[[TC]]/../../../../sysroot/micromips-r6-hard"
// EB-HARD-MICRO: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EB-HARD-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-HARD-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-HARD-MICRO: "[[TC]]/micromips-r6-hard/lib{{/|\\\\}}crtbegin.o"
// EB-HARD-MICRO: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/micromips-r6-hard/lib"
// EB-HARD-MICRO: "-L[[TC]]/micromips-r6-hard/lib"
// EB-HARD-MICRO: "-L[[TC]]/../../../../sysroot/micromips-r6-hard/lib/../lib"
// EB-HARD-MICRO: "-L[[TC]]/../../../../sysroot/micromips-r6-hard/usr/lib/../lib"
// EB-HARD-MICRO: "[[TC]]/micromips-r6-hard/lib{{/|\\\\}}crtend.o"
// EB-HARD-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EB -mips32r6 -msoft-float -mmicromips
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EB -mips32r6 -msoft-float -mmicromips -no-pie \
// RUN:   | FileCheck --check-prefix=EB-SOFT-MICRO %s
// EB-SOFT-MICRO: "-internal-isystem"
// EB-SOFT-MICRO: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EB-SOFT-MICRO: "-internal-isystem"
// EB-SOFT-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/micromips-r6-soft/lib"
// EB-SOFT-MICRO: "-internal-isystem"
// EB-SOFT-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EB-SOFT-MICRO: "-internal-externc-isystem"
// EB-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-soft/lib/../usr/include"
// EB-SOFT-MICRO: "{{.*}}ld{{(.exe)?}}"
// EB-SOFT-MICRO: "--sysroot=[[TC]]/../../../../sysroot/micromips-r6-soft"
// EB-SOFT-MICRO: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EB-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EB-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EB-SOFT-MICRO: "[[TC]]/micromips-r6-soft/lib{{/|\\\\}}crtbegin.o"
// EB-SOFT-MICRO: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/micromips-r6-soft/lib"
// EB-SOFT-MICRO: "-L[[TC]]/micromips-r6-soft/lib"
// EB-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromips-r6-soft/lib/../lib"
// EB-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromips-r6-soft/usr/lib/../lib"
// EB-SOFT-MICRO: "[[TC]]/micromips-r6-soft/lib{{/|\\\\}}crtend.o"
// EB-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromips-r6-soft/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mips32r6 -mhard-float -mmicromips
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips32r6 -mhard-float -mmicromips -no-pie \
// RUN:   | FileCheck --check-prefix=EL-HARD-MICRO %s
// EL-HARD-MICRO: "-internal-isystem"
// EL-HARD-MICRO: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-HARD-MICRO: "-internal-isystem"
// EL-HARD-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/micromipsel-r6-hard/lib"
// EL-HARD-MICRO: "-internal-isystem"
// EL-HARD-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-HARD-MICRO: "-internal-externc-isystem"
// EL-HARD-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-hard/lib/../usr/include"
// EL-HARD-MICRO: "{{.*}}ld{{(.exe)?}}"
// EL-HARD-MICRO: "--sysroot=[[TC]]/../../../../sysroot/micromipsel-r6-hard"
// EL-HARD-MICRO: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-HARD-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-HARD-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-HARD-MICRO: "[[TC]]/micromipsel-r6-hard/lib{{/|\\\\}}crtbegin.o"
// EL-HARD-MICRO: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/micromipsel-r6-hard/lib"
// EL-HARD-MICRO: "-L[[TC]]/micromipsel-r6-hard/lib"
// EL-HARD-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r6-hard/lib/../lib"
// EL-HARD-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r6-hard/usr/lib/../lib"
// EL-HARD-MICRO: "[[TC]]/micromipsel-r6-hard/lib{{/|\\\\}}crtend.o"
// EL-HARD-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-hard/usr/lib/../lib{{/|\\\\}}crtn.o"

// -EL -mips32r6 -msoft-float -mmicromips
// RUN: %clang -### %s 2>&1 \
// RUN:        --target=mips-img-linux-gnu \
// RUN:        --gcc-toolchain=%S/Inputs/mips_img_v2_tree \
// RUN:        -stdlib=libstdc++ \
// RUN:        -EL -mips32r6 -msoft-float -mmicromips -no-pie \
// RUN:   | FileCheck --check-prefix=EL-SOFT-MICRO %s
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.2]]/../../../../mips-img-linux-gnu/include/c++/4.9.2"
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/mips-img-linux-gnu/micromipsel-r6-soft/lib"
// EL-SOFT-MICRO: "-internal-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.2/backward"
// EL-SOFT-MICRO: "-internal-externc-isystem"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-soft/lib/../usr/include"
// EL-SOFT-MICRO: "{{.*}}ld{{(.exe)?}}"
// EL-SOFT-MICRO: "--sysroot=[[TC]]/../../../../sysroot/micromipsel-r6-soft"
// EL-SOFT-MICRO: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crt1.o"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crti.o"
// EL-SOFT-MICRO: "[[TC]]/micromipsel-r6-soft/lib{{/|\\\\}}crtbegin.o"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/micromipsel-r6-soft/lib"
// EL-SOFT-MICRO: "-L[[TC]]/micromipsel-r6-soft/lib"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r6-soft/lib/../lib"
// EL-SOFT-MICRO: "-L[[TC]]/../../../../sysroot/micromipsel-r6-soft/usr/lib/../lib"
// EL-SOFT-MICRO: "[[TC]]/micromipsel-r6-soft/lib{{/|\\\\}}crtend.o"
// EL-SOFT-MICRO: "[[TC]]/../../../../sysroot/micromipsel-r6-soft/usr/lib/../lib{{/|\\\\}}crtn.o"
