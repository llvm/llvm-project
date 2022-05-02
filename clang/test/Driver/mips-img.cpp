// REQUIRES: mips-registered-target

// Check frontend and linker invocations on the IMG MIPS toolchain.
//
// = Big-endian, mips32r6
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips-img-linux-gnu -mips32r6 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-32R6 %s
// CHECK-BE-32R6: "-internal-isystem"
// CHECK-BE-32R6: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-BE-32R6: "-internal-isystem"
// CHECK-BE-32R6: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu"
// CHECK-BE-32R6: "-internal-isystem"
// CHECK-BE-32R6: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-32R6: "-internal-externc-isystem"
// CHECK-BE-32R6: "[[TC]]/include"
// CHECK-BE-32R6: "-internal-externc-isystem"
// CHECK-BE-32R6: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-32R6: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-32R6: "--sysroot=[[TC]]/../../../../sysroot"
// CHECK-BE-32R6: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-32R6: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-32R6: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-32R6: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-32R6: "-L[[TC]]"
// CHECK-BE-32R6: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/../lib"
// CHECK-BE-32R6: "-L[[TC]]/../../../../sysroot/usr/lib/../lib"
// CHECK-BE-32R6: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-32R6: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r6
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips-img-linux-gnu -mips32r6 -EL -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LE-32R6 %s
// CHECK-LE-32R6: "-internal-isystem"
// CHECK-LE-32R6: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-LE-32R6: "-internal-isystem"
// CHECK-LE-32R6: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu/el"
// CHECK-LE-32R6: "-internal-isystem"
// CHECK-LE-32R6: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-LE-32R6: "-internal-externc-isystem"
// CHECK-LE-32R6: "[[TC]]/include"
// CHECK-LE-32R6: "-internal-externc-isystem"
// CHECK-LE-32R6: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-LE-32R6: "{{.*}}ld{{(.exe)?}}"
// CHECK-LE-32R6: "--sysroot=[[TC]]/../../../../sysroot/el"
// CHECK-LE-32R6: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-LE-32R6: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-LE-32R6: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-LE-32R6: "[[TC]]/el{{/|\\\\}}crtbegin.o"
// CHECK-LE-32R6: "-L[[TC]]/el"
// CHECK-LE-32R6: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/../lib/el"
// CHECK-LE-32R6: "-L[[TC]]/../../../../sysroot/el/usr/lib/../lib"
// CHECK-LE-32R6: "[[TC]]/el{{/|\\\\}}crtend.o"
// CHECK-LE-32R6: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r6, N32
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips64-img-linux-gnu -mips64r6 -mabi=n32 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-64R6-N32 %s
// CHECK-BE-64R6-N32: "-internal-isystem"
// CHECK-BE-64R6-N32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-BE-64R6-N32: "-internal-isystem"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu/mips64r6"
// CHECK-BE-64R6-N32: "-internal-isystem"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-64R6-N32: "-internal-externc-isystem"
// CHECK-BE-64R6-N32: "[[TC]]/include"
// CHECK-BE-64R6-N32: "-internal-externc-isystem"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-64R6-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-64R6-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r6"
// CHECK-BE-64R6-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-64R6-N32: "[[TC]]/mips64r6{{/|\\\\}}crtbegin.o"
// CHECK-BE-64R6-N32: "-L[[TC]]/mips64r6"
// CHECK-BE-64R6-N32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips64r6"
// CHECK-BE-64R6-N32: "-L[[TC]]/../../../../sysroot/mips64r6/usr/lib"
// CHECK-BE-64R6-N32: "[[TC]]/mips64r6{{/|\\\\}}crtend.o"
// CHECK-BE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r6, N32
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips64-img-linux-gnu -mips64r6 -EL -mabi=n32 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LE-64R6-N32 %s
// CHECK-LE-64R6-N32: "-internal-isystem"
// CHECK-LE-64R6-N32: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-LE-64R6-N32: "-internal-isystem"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu/mips64r6/el"
// CHECK-LE-64R6-N32: "-internal-isystem"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-LE-64R6-N32: "-internal-externc-isystem"
// CHECK-LE-64R6-N32: "[[TC]]/include"
// CHECK-LE-64R6-N32: "-internal-externc-isystem"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-LE-64R6-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LE-64R6-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r6/el"
// CHECK-LE-64R6-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-LE-64R6-N32: "[[TC]]/mips64r6/el{{/|\\\\}}crtbegin.o"
// CHECK-LE-64R6-N32: "-L[[TC]]/mips64r6/el"
// CHECK-LE-64R6-N32: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips64r6/el"
// CHECK-LE-64R6-N32: "-L[[TC]]/../../../../sysroot/mips64r6/el/usr/lib"
// CHECK-LE-64R6-N32: "[[TC]]/mips64r6/el{{/|\\\\}}crtend.o"
// CHECK-LE-64R6-N32: "[[TC]]/../../../../sysroot/mips64r6/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r6, N64
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips64-img-linux-gnu -mips64r6 -mabi=64 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-64R6-N64 %s
// CHECK-BE-64R6-N64: "-internal-isystem"
// CHECK-BE-64R6-N64: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-BE-64R6-N64: "-internal-isystem"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu/mips64r6/64"
// CHECK-BE-64R6-N64: "-internal-isystem"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-64R6-N64: "-internal-externc-isystem"
// CHECK-BE-64R6-N64: "[[TC]]/include"
// CHECK-BE-64R6-N64: "-internal-externc-isystem"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-64R6-N64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-64R6-N64: "--sysroot=[[TC]]/../../../../sysroot/mips64r6/64"
// CHECK-BE-64R6-N64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-64R6-N64: "[[TC]]/mips64r6/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-64R6-N64: "-L[[TC]]/mips64r6/64"
// CHECK-BE-64R6-N64: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips64r6/64"
// CHECK-BE-64R6-N64: "-L[[TC]]/../../../../sysroot/mips64r6/64/usr/lib"
// CHECK-BE-64R6-N64: "[[TC]]/mips64r6/64{{/|\\\\}}crtend.o"
// CHECK-BE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r6, N64
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=mips64-img-linux-gnu -mips64r6 -EL -mabi=64 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_img_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LE-64R6-N64 %s
// CHECK-LE-64R6-N64: "-internal-isystem"
// CHECK-LE-64R6-N64: "[[TC:[^"]+/lib/gcc/mips-img-linux-gnu/4.9.0]]/../../../../mips-img-linux-gnu/include/c++/4.9.0"
// CHECK-LE-64R6-N64: "-internal-isystem"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/mips-img-linux-gnu/mips64r6/64/el"
// CHECK-LE-64R6-N64: "-internal-isystem"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../mips-img-linux-gnu/include/c++/4.9.0/backward"
// CHECK-LE-64R6-N64: "-internal-externc-isystem"
// CHECK-LE-64R6-N64: "[[TC]]/include"
// CHECK-LE-64R6-N64: "-internal-externc-isystem"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-LE-64R6-N64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LE-64R6-N64: "--sysroot=[[TC]]/../../../../sysroot/mips64r6/64/el"
// CHECK-LE-64R6-N64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-LE-64R6-N64: "[[TC]]/mips64r6/64/el{{/|\\\\}}crtbegin.o"
// CHECK-LE-64R6-N64: "-L[[TC]]/mips64r6/64/el"
// CHECK-LE-64R6-N64: "-L[[TC]]/../../../../mips-img-linux-gnu/lib/mips64r6/64/el"
// CHECK-LE-64R6-N64: "-L[[TC]]/../../../../sysroot/mips64r6/64/el/usr/lib"
// CHECK-LE-64R6-N64: "[[TC]]/mips64r6/64/el{{/|\\\\}}crtend.o"
// CHECK-LE-64R6-N64: "[[TC]]/../../../../sysroot/mips64r6/64/el/usr/lib{{/|\\\\}}crtn.o"
