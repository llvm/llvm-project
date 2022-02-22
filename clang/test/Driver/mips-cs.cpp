// REQUIRES: mips-registered-target
//
// Check frontend and linker invocations on Mentor Graphics MIPS toolchain.
//
// = Big-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32 %s
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/include"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc"
// CHECK-BE-HF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-32: "-L[[TC]]"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/lib"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/usr/lib"
// CHECK-BE-HF-32: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -muclibc -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-HF-32 %s
// CHECK-BE-UC-HF-32: "-internal-isystem"
// CHECK-BE-UC-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-UC-HF-32: "-internal-isystem"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc"
// CHECK-BE-UC-HF-32: "-internal-isystem"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-UC-HF-32: "-internal-externc-isystem"
// CHECK-BE-UC-HF-32: "[[TC]]/include"
// CHECK-BE-UC-HF-32: "-internal-externc-isystem"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-BE-UC-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-HF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc"
// CHECK-BE-UC-HF-32: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-HF-32: "[[TC]]/uclibc{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-HF-32: "-L[[TC]]/uclibc"
// CHECK-BE-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc"
// CHECK-BE-UC-HF-32-NOT: "-L[[TC]]"
// CHECK-BE-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/lib/../lib"
// CHECK-BE-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/lib/../lib"
// CHECK-BE-UC-HF-32: "[[TC]]/uclibc{{/|\\\\}}crtend.o"
// CHECK-BE-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -mips16 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-16 %s
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/include"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-16: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/mips16"
// CHECK-BE-HF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-16: "[[TC]]/mips16{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-16: "-L[[TC]]/mips16"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16"
// CHECK-BE-HF-16-NOT: "-L[[TC]]"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/lib/../lib"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib"
// CHECK-BE-HF-16: "[[TC]]/mips16{{/|\\\\}}crtend.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, mmicromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -mmicromips -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-MICRO %s
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips"
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-MICRO: "-internal-externc-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/include"
// CHECK-BE-HF-MICRO: "-internal-externc-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-HF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-MICRO: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/micromips"
// CHECK-BE-HF-MICRO: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-MICRO: "[[TC]]/micromips{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-MICRO: "-L[[TC]]/micromips"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips"
// CHECK-BE-HF-MICRO-NOT: "-L[[TC]]"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/lib/../lib"
// CHECK-BE-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib"
// CHECK-BE-HF-MICRO: "[[TC]]/micromips{{/|\\\\}}crtend.o"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -mnan=2008 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-NAN %s
// CHECK-BE-HF-NAN: "-internal-isystem"
// CHECK-BE-HF-NAN: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-NAN: "-internal-isystem"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/nan2008"
// CHECK-BE-HF-NAN: "-internal-isystem"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-NAN: "-internal-externc-isystem"
// CHECK-BE-HF-NAN: "[[TC]]/include"
// CHECK-BE-HF-NAN: "-internal-externc-isystem"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-HF-NAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-NAN: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/nan2008"
// CHECK-BE-HF-NAN: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-NAN: "[[TC]]/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-NAN: "-L[[TC]]/nan2008"
// CHECK-BE-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/nan2008"
// CHECK-BE-HF-NAN-NOT: "-L[[TC]]"
// CHECK-BE-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/nan2008/lib/../lib"
// CHECK-BE-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/nan2008/usr/lib/../lib"
// CHECK-BE-HF-NAN: "[[TC]]/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, uclibc, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -muclibc -mnan=2008 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-HF-NAN %s
// CHECK-BE-UC-HF-NAN: "-internal-isystem"
// CHECK-BE-UC-HF-NAN: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-UC-HF-NAN: "-internal-isystem"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc/nan2008"
// CHECK-BE-UC-HF-NAN: "-internal-isystem"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-UC-HF-NAN: "-internal-externc-isystem"
// CHECK-BE-UC-HF-NAN: "[[TC]]/include"
// CHECK-BE-UC-HF-NAN: "-internal-externc-isystem"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-BE-UC-HF-NAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-HF-NAN: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008"
// CHECK-BE-UC-HF-NAN: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-HF-NAN: "[[TC]]/uclibc/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-HF-NAN: "-L[[TC]]/uclibc/nan2008"
// CHECK-BE-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc/nan2008"
// CHECK-BE-UC-HF-NAN-NOT: "-L[[TC]]"
// CHECK-BE-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/lib/../lib"
// CHECK-BE-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/usr/lib/../lib"
// CHECK-BE-UC-HF-NAN: "[[TC]]/uclibc/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-32 %s
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/include"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/soft-float"
// CHECK-BE-SF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-32: "[[TC]]/soft-float{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-32: "-L[[TC]]/soft-float"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/soft-float"
// CHECK-BE-SF-32-NOT: "-L[[TC]]"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/lib/../lib"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib"
// CHECK-BE-SF-32: "[[TC]]/soft-float{{/|\\\\}}crtend.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, soft float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -muclibc -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-SF-32 %s
// CHECK-BE-UC-SF-32: "-internal-isystem"
// CHECK-BE-UC-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-UC-SF-32: "-internal-isystem"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc/soft-float"
// CHECK-BE-UC-SF-32: "-internal-isystem"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-UC-SF-32: "-internal-externc-isystem"
// CHECK-BE-UC-SF-32: "[[TC]]/include"
// CHECK-BE-UC-SF-32: "-internal-externc-isystem"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-BE-UC-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-SF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float"
// CHECK-BE-UC-SF-32: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-SF-32: "[[TC]]/uclibc/soft-float{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-SF-32: "-L[[TC]]/uclibc/soft-float"
// CHECK-BE-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc/soft-float"
// CHECK-BE-UC-SF-32-NOT: "-L[[TC]]"
// CHECK-BE-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/lib/../lib"
// CHECK-BE-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/usr/lib/../lib"
// CHECK-BE-UC-SF-32: "[[TC]]/uclibc/soft-float{{/|\\\\}}crtend.o"
// CHECK-BE-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -msoft-float -mips16 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-16 %s
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/soft-float"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/include"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-16: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float"
// CHECK-BE-SF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-16: "[[TC]]/mips16/soft-float{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-16: "-L[[TC]]/mips16/soft-float"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/soft-float"
// CHECK-BE-SF-16-NOT: "-L[[TC]]"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/lib/../lib"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib"
// CHECK-BE-SF-16: "[[TC]]/mips16/soft-float{{/|\\\\}}crtend.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu -msoft-float -mmicromips -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-MICRO %s
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-MICRO: "-internal-externc-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/include"
// CHECK-BE-SF-MICRO: "-internal-externc-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-SF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-MICRO: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-MICRO: "[[TC]]/micromips/soft-float{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-MICRO: "-L[[TC]]/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/soft-float"
// CHECK-BE-SF-MICRO-NOT: "-L[[TC]]"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/lib/../lib"
// CHECK-BE-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib"
// CHECK-BE-SF-MICRO: "[[TC]]/micromips/soft-float{{/|\\\\}}crtend.o"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64 %s
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/64"
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-64: "-internal-externc-isystem"
// CHECK-BE-HF-64: "[[TC]]/include"
// CHECK-BE-HF-64: "-internal-externc-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-HF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc"
// CHECK-BE-HF-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64{{/|\\\\}}crti.o"
// CHECK-BE-HF-64: "[[TC]]/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64: "-L[[TC]]/64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/lib/../lib64"
// CHECK-BE-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64"
// CHECK-BE-HF-64-NOT: "-L[[TC]]"
// CHECK-BE-HF-64: "{{.*}}/lib/gcc/mips-linux-gnu/4.6.3/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/lib/../lib64{{/|\\\\}}crtn.o"
//
// = Big-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64 %s
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/64"
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-64: "-internal-externc-isystem"
// CHECK-BE-SF-64: "[[TC]]/include"
// CHECK-BE-SF-64: "-internal-externc-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-BE-SF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/soft-float"
// CHECK-BE-SF-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64{{/|\\\\}}crt1.o"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64{{/|\\\\}}crti.o"
// CHECK-BE-SF-64: "[[TC]]/soft-float/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-64: "-L[[TC]]/soft-float/64"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/soft-float"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/lib/../lib64"
// CHECK-BE-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64"
// CHECK-BE-SF-64-NOT: "-L[[TC]]"
// CHECK-BE-SF-64: "[[TC]]/soft-float/64{{/|\\\\}}crtend.o"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/usr/lib/../lib64{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mhard-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-32 %s
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/el"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/include"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/el"
// CHECK-EL-HF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-32: "[[TC]]/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-32: "-L[[TC]]/el"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/el"
// CHECK-EL-HF-32-NOT: "-L[[TC]]"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/lib/../lib"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib"
// CHECK-EL-HF-32: "[[TC]]/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mhard-float -muclibc -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-HF-32 %s
// CHECK-EL-UC-HF-32: "-internal-isystem"
// CHECK-EL-UC-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-UC-HF-32: "-internal-isystem"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc/el"
// CHECK-EL-UC-HF-32: "-internal-isystem"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-UC-HF-32: "-internal-externc-isystem"
// CHECK-EL-UC-HF-32: "[[TC]]/include"
// CHECK-EL-UC-HF-32: "-internal-externc-isystem"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-EL-UC-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-HF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el"
// CHECK-EL-UC-HF-32: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-HF-32: "[[TC]]/uclibc/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-HF-32: "-L[[TC]]/uclibc/el"
// CHECK-EL-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc/el"
// CHECK-EL-UC-HF-32-NOT: "-L[[TC]]"
// CHECK-EL-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el/lib/../lib"
// CHECK-EL-UC-HF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el/usr/lib/../lib"
// CHECK-EL-UC-HF-32: "[[TC]]/uclibc/el{{/|\\\\}}crtend.o"
// CHECK-EL-UC-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mips16 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-16 %s
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/el"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/include"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-16: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/mips16/el"
// CHECK-EL-HF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-16: "[[TC]]/mips16/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-16: "-L[[TC]]/mips16/el"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/el"
// CHECK-EL-HF-16-NOT: "-L[[TC]]"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/lib/../lib"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib"
// CHECK-EL-HF-16: "[[TC]]/mips16/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mmicromips -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-MICRO %s
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/el"
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-MICRO: "-internal-externc-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/include"
// CHECK-EL-HF-MICRO: "-internal-externc-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-HF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-MICRO: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/micromips/el"
// CHECK-EL-HF-MICRO: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-MICRO: "[[TC]]/micromips/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-MICRO: "-L[[TC]]/micromips/el"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/el"
// CHECK-EL-HF-MICRO-NOT: "-L[[TC]]"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/lib/../lib"
// CHECK-EL-HF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib"
// CHECK-EL-HF-MICRO: "[[TC]]/micromips/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mnan=2008 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-NAN %s
// CHECK-EL-HF-NAN: "-internal-isystem"
// CHECK-EL-HF-NAN: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-NAN: "-internal-isystem"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/nan2008/el"
// CHECK-EL-HF-NAN: "-internal-isystem"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-NAN: "-internal-externc-isystem"
// CHECK-EL-HF-NAN: "[[TC]]/include"
// CHECK-EL-HF-NAN: "-internal-externc-isystem"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-HF-NAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-NAN: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el"
// CHECK-EL-HF-NAN: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-NAN: "[[TC]]/nan2008/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-NAN: "-L[[TC]]/nan2008/el"
// CHECK-EL-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/nan2008/el"
// CHECK-EL-HF-NAN-NOT: "-L[[TC]]"
// CHECK-EL-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el/lib/../lib"
// CHECK-EL-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el/usr/lib/../lib"
// CHECK-EL-HF-NAN: "[[TC]]/nan2008/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/nan2008/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, uclibc, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -muclibc -mnan=2008 -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-HF-NAN %s
// CHECK-EL-UC-HF-NAN: "-internal-isystem"
// CHECK-EL-UC-HF-NAN: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-UC-HF-NAN: "-internal-isystem"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc/nan2008/el"
// CHECK-EL-UC-HF-NAN: "-internal-isystem"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-UC-HF-NAN: "-internal-externc-isystem"
// CHECK-EL-UC-HF-NAN: "[[TC]]/include"
// CHECK-EL-UC-HF-NAN: "-internal-externc-isystem"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-EL-UC-HF-NAN: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-HF-NAN: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el"
// CHECK-EL-UC-HF-NAN: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-HF-NAN: "[[TC]]/uclibc/nan2008/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-HF-NAN: "-L[[TC]]/uclibc/nan2008/el"
// CHECK-EL-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc/nan2008/el"
// CHECK-EL-UC-HF-NAN-NOT: "-L[[TC]]"
// CHECK-EL-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el/lib/../lib"
// CHECK-EL-UC-HF-NAN: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el/usr/lib/../lib"
// CHECK-EL-UC-HF-NAN: "[[TC]]/uclibc/nan2008/el{{/|\\\\}}crtend.o"
// CHECK-EL-UC-HF-NAN: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/nan2008/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mfloat-abi=soft -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-32 %s
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/el"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/include"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el"
// CHECK-EL-SF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-32: "[[TC]]/soft-float/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-32: "-L[[TC]]/soft-float/el"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/soft-float/el"
// CHECK-EL-SF-32-NOT: "-L[[TC]]"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/lib/../lib"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib"
// CHECK-EL-SF-32: "[[TC]]/soft-float/el{{/|\\\\}}crtend.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, soft float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mfloat-abi=soft -muclibc -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-SF-32 %s
// CHECK-EL-UC-SF-32: "-internal-isystem"
// CHECK-EL-UC-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-UC-SF-32: "-internal-isystem"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/uclibc/soft-float/el"
// CHECK-EL-UC-SF-32: "-internal-isystem"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-UC-SF-32: "-internal-externc-isystem"
// CHECK-EL-UC-SF-32: "[[TC]]/include"
// CHECK-EL-UC-SF-32: "-internal-externc-isystem"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/usr/include"
// CHECK-EL-UC-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-SF-32: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el"
// CHECK-EL-UC-SF-32: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-SF-32: "[[TC]]/uclibc/soft-float/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-SF-32: "-L[[TC]]/uclibc/soft-float/el"
// CHECK-EL-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/uclibc/soft-float/el"
// CHECK-EL-UC-SF-32-NOT: "-L[[TC]]"
// CHECK-EL-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el/lib/../lib"
// CHECK-EL-UC-SF-32: "-L[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el/usr/lib/../lib"
// CHECK-EL-UC-SF-32: "[[TC]]/uclibc/soft-float/el{{/|\\\\}}crtend.o"
// CHECK-EL-UC-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/uclibc/soft-float/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mips16 -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-16 %s
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/soft-float/el"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/include"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-16: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el"
// CHECK-EL-SF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-16: "[[TC]]/mips16/soft-float/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-16: "-L[[TC]]/mips16/soft-float/el"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/mips16/soft-float/el"
// CHECK-EL-SF-16-NOT: "-L[[TC]]"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/lib/../lib"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib"
// CHECK-EL-SF-16: "[[TC]]/mips16/soft-float/el{{/|\\\\}}crtend.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/mips16/soft-float/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -mmicromips -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-MICRO %s
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-MICRO: "-internal-externc-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/include"
// CHECK-EL-SF-MICRO: "-internal-externc-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-SF-MICRO: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-MICRO: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-MICRO: "[[TC]]/micromips/soft-float/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-MICRO: "-L[[TC]]/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib/micromips/soft-float/el"
// CHECK-EL-SF-MICRO-NOT: "-L[[TC]]"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/lib/../lib"
// CHECK-EL-SF-MICRO: "-L[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib"
// CHECK-EL-SF-MICRO: "[[TC]]/micromips/soft-float/el{{/|\\\\}}crtend.o"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/micromips/soft-float/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64 %s
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/el/64"
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-64: "-internal-externc-isystem"
// CHECK-EL-HF-64: "[[TC]]/include"
// CHECK-EL-HF-64: "-internal-externc-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-HF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/el"
// CHECK-EL-HF-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64{{/|\\\\}}crt1.o"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64{{/|\\\\}}crti.o"
// CHECK-EL-HF-64: "[[TC]]/el/64{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-64: "-L[[TC]]/el/64"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/el"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/lib/../lib64"
// CHECK-EL-HF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64"
// CHECK-EL-HF-64-NOT: "-L[[TC]]"
// CHECK-EL-HF-64: "[[TC]]/el/64{{/|\\\\}}crtend.o"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/el/usr/lib/../lib64{{/|\\\\}}crtn.o"
//
// = Little-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -msoft-float -no-pie \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64 %s
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/el/64"
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-64: "-internal-externc-isystem"
// CHECK-EL-SF-64: "[[TC]]/include"
// CHECK-EL-SF-64: "-internal-externc-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
// CHECK-EL-SF-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64: "--sysroot=[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el"
// CHECK-EL-SF-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64{{/|\\\\}}crt1.o"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64{{/|\\\\}}crti.o"
// CHECK-EL-SF-64: "[[TC]]/soft-float/el/64{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-64: "-L[[TC]]/soft-float/el/64"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/lib/../lib64/soft-float/el"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/lib/../lib64"
// CHECK-EL-SF-64: "-L[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64"
// CHECK-EL-SF-64-NOT: "-L[[TC]]"
// CHECK-EL-SF-64: "[[TC]]/soft-float/el/64{{/|\\\\}}crtend.o"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/soft-float/el/usr/lib/../lib64{{/|\\\\}}crtn.o"
