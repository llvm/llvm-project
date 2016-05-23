// Check frontend and linker invocations on FSF MIPS toolchain.
//
// = Big-endian, mips32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32 %s
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/include"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32: "--sysroot=[[TC]]/../../../../sysroot/mips32"
// CHECK-BE-HF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32: "[[TC]]/mips32{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-32: "-L[[TC]]/mips32"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32"
// CHECK-BE-HF-32: "-L[[TC]]/../../../../sysroot/mips32/usr/lib/../lib"
// CHECK-BE-HF-32: "[[TC]]/mips32{{/|\\\\}}crtend.o"
// CHECK-BE-HF-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32, hard float, fp64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-32 %s
// CHECK-BE-HF64-32: "-internal-isystem"
// CHECK-BE-HF64-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-32: "-internal-isystem"
// CHECK-BE-HF64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32"
// CHECK-BE-HF64-32: "-internal-isystem"
// CHECK-BE-HF64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-32: "-internal-externc-isystem"
// CHECK-BE-HF64-32: "[[TC]]/include"
// CHECK-BE-HF64-32: "-internal-externc-isystem"
// CHECK-BE-HF64-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-32: "--sysroot=[[TC]]/../../../../sysroot/mips32"
// CHECK-BE-HF64-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF64-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-32: "[[TC]]/mips32{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-32: "-L[[TC]]/mips32"
// CHECK-BE-HF64-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32"
// CHECK-BE-HF64-32: "-L[[TC]]/../../../../sysroot/mips32/usr/lib/../lib"
// CHECK-BE-HF64-32: "[[TC]]/mips32{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-32: "[[TC]]/../../../../sysroot/mips32/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-32 %s
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/sof"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/include"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/sof"
// CHECK-BE-SF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-32: "[[TC]]/../../../../sysroot/mips32/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../sysroot/mips32/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-32: "[[TC]]/mips32/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-32: "-L[[TC]]/mips32/sof"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/sof"
// CHECK-BE-SF-32: "-L[[TC]]/../../../../sysroot/mips32/sof/usr/lib/../lib"
// CHECK-BE-SF-32: "[[TC]]/mips32/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-32: "[[TC]]/../../../../sysroot/mips32/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips16 / mips32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mips16 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-16 %s
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/include"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16"
// CHECK-BE-HF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-16: "[[TC]]/mips32/mips16{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-16: "-L[[TC]]/mips32/mips16"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16"
// CHECK-BE-HF-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib"
// CHECK-BE-HF-16: "[[TC]]/mips32/mips16{{/|\\\\}}crtend.o"
// CHECK-BE-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips16 / mips32, hard float, fp64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mips16 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-16 %s
// CHECK-BE-HF64-16: "-internal-isystem"
// CHECK-BE-HF64-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-16: "-internal-isystem"
// CHECK-BE-HF64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16"
// CHECK-BE-HF64-16: "-internal-isystem"
// CHECK-BE-HF64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-16: "-internal-externc-isystem"
// CHECK-BE-HF64-16: "[[TC]]/include"
// CHECK-BE-HF64-16: "-internal-externc-isystem"
// CHECK-BE-HF64-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16"
// CHECK-BE-HF64-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-16: "[[TC]]/mips32/mips16{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-16: "-L[[TC]]/mips32/mips16"
// CHECK-BE-HF64-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16"
// CHECK-BE-HF64-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib"
// CHECK-BE-HF64-16: "[[TC]]/mips32/mips16{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips16 / mips32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mips16 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-16 %s
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/sof"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/include"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/sof"
// CHECK-BE-SF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-16: "[[TC]]/mips32/mips16/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-16: "-L[[TC]]/mips32/mips16/sof"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/sof"
// CHECK-BE-SF-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/sof/usr/lib/../lib"
// CHECK-BE-SF-16: "[[TC]]/mips32/mips16/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32 / mips16, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mips16 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-16 %s
// CHECK-BE-NAN-16: "-internal-isystem"
// CHECK-BE-NAN-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-16: "-internal-isystem"
// CHECK-BE-NAN-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/nan2008"
// CHECK-BE-NAN-16: "-internal-isystem"
// CHECK-BE-NAN-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-16: "-internal-externc-isystem"
// CHECK-BE-NAN-16: "[[TC]]/include"
// CHECK-BE-NAN-16: "-internal-externc-isystem"
// CHECK-BE-NAN-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/nan2008"
// CHECK-BE-NAN-16: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-16: "[[TC]]/mips32/mips16/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-16: "-L[[TC]]/mips32/mips16/nan2008"
// CHECK-BE-NAN-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/nan2008"
// CHECK-BE-NAN-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib"
// CHECK-BE-NAN-16: "[[TC]]/mips32/mips16/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32 / mips16, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mips16 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-16 %s
// CHECK-BE-NAN64-16: "-internal-isystem"
// CHECK-BE-NAN64-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-16: "-internal-isystem"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/nan2008"
// CHECK-BE-NAN64-16: "-internal-isystem"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-16: "-internal-externc-isystem"
// CHECK-BE-NAN64-16: "[[TC]]/include"
// CHECK-BE-NAN64-16: "-internal-externc-isystem"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/nan2008"
// CHECK-BE-NAN64-16: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-16: "[[TC]]/mips32/mips16/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-16: "-L[[TC]]/mips32/mips16/nan2008"
// CHECK-BE-NAN64-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/nan2008"
// CHECK-BE-NAN64-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-16: "[[TC]]/mips32/mips16/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-32 %s
// CHECK-BE-NAN-32: "-internal-isystem"
// CHECK-BE-NAN-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-32: "-internal-isystem"
// CHECK-BE-NAN-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/nan2008"
// CHECK-BE-NAN-32: "-internal-isystem"
// CHECK-BE-NAN-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-32: "-internal-externc-isystem"
// CHECK-BE-NAN-32: "[[TC]]/include"
// CHECK-BE-NAN-32: "-internal-externc-isystem"
// CHECK-BE-NAN-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/nan2008"
// CHECK-BE-NAN-32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-32: "[[TC]]/mips32/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-32: "-L[[TC]]/mips32/nan2008"
// CHECK-BE-NAN-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/nan2008"
// CHECK-BE-NAN-32: "-L[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib"
// CHECK-BE-NAN-32: "[[TC]]/mips32/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-32 %s
// CHECK-BE-NAN64-32: "-internal-isystem"
// CHECK-BE-NAN64-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-32: "-internal-isystem"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/nan2008"
// CHECK-BE-NAN64-32: "-internal-isystem"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-32: "-internal-externc-isystem"
// CHECK-BE-NAN64-32: "[[TC]]/include"
// CHECK-BE-NAN64-32: "-internal-externc-isystem"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/nan2008"
// CHECK-BE-NAN64-32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-32: "[[TC]]/mips32/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-32: "-L[[TC]]/mips32/nan2008"
// CHECK-BE-NAN64-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/nan2008"
// CHECK-BE-NAN64-32: "-L[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-32: "[[TC]]/mips32/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-32: "[[TC]]/../../../../sysroot/mips32/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32R2 %s
// CHECK-BE-HF-32R2: "-internal-isystem"
// CHECK-BE-HF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-32R2: "-internal-isystem"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu"
// CHECK-BE-HF-32R2: "-internal-isystem"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-32R2: "-internal-externc-isystem"
// CHECK-BE-HF-32R2: "[[TC]]/include"
// CHECK-BE-HF-32R2: "-internal-externc-isystem"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32R2: "--sysroot=[[TC]]/../../../../sysroot"
// CHECK-BE-HF-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32R2: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-32R2: "-L[[TC]]"
// CHECK-BE-HF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib"
// CHECK-BE-HF-32R2: "-L[[TC]]/../../../../sysroot/usr/lib/../lib"
// CHECK-BE-HF-32R2: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-HF-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, hard float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mhard-float -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-HF-32R2 %s
// CHECK-BE-UC-HF-32R2: "-internal-isystem"
// CHECK-BE-UC-HF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-UC-HF-32R2: "-internal-isystem"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc"
// CHECK-BE-UC-HF-32R2: "-internal-isystem"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-UC-HF-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-HF-32R2: "[[TC]]/include"
// CHECK-BE-UC-HF-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-BE-UC-HF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-HF-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc"
// CHECK-BE-UC-HF-32R2: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-HF-32R2: "[[TC]]/uclibc{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-HF-32R2: "-L[[TC]]/uclibc"
// CHECK-BE-UC-HF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc"
// CHECK-BE-UC-HF-32R2: "-L[[TC]]/../../../../sysroot/uclibc/usr/lib/../lib"
// CHECK-BE-UC-HF-32R2: "[[TC]]/uclibc{{/|\\\\}}crtend.o"
// CHECK-BE-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-32R2 %s
// CHECK-BE-HF64-32R2: "-internal-isystem"
// CHECK-BE-HF64-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-32R2: "-internal-isystem"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu"
// CHECK-BE-HF64-32R2: "-internal-isystem"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-32R2: "-internal-externc-isystem"
// CHECK-BE-HF64-32R2: "[[TC]]/include"
// CHECK-BE-HF64-32R2: "-internal-externc-isystem"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-32R2: "--sysroot=[[TC]]/../../../../sysroot"
// CHECK-BE-HF64-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-32R2: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-32R2: "-L[[TC]]"
// CHECK-BE-HF64-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib"
// CHECK-BE-HF64-32R2: "-L[[TC]]/../../../../sysroot/usr/lib/../lib"
// CHECK-BE-HF64-32R2: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-32R2: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-32R2 %s
// CHECK-BE-SF-32R2: "-internal-isystem"
// CHECK-BE-SF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-32R2: "-internal-isystem"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/sof"
// CHECK-BE-SF-32R2: "-internal-isystem"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-32R2: "-internal-externc-isystem"
// CHECK-BE-SF-32R2: "[[TC]]/include"
// CHECK-BE-SF-32R2: "-internal-externc-isystem"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-32R2: "--sysroot=[[TC]]/../../../../sysroot/sof"
// CHECK-BE-SF-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../sysroot/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../sysroot/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-32R2: "[[TC]]/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-32R2: "-L[[TC]]/sof"
// CHECK-BE-SF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/sof"
// CHECK-BE-SF-32R2: "-L[[TC]]/../../../../sysroot/sof/usr/lib/../lib"
// CHECK-BE-SF-32R2: "[[TC]]/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-32R2: "[[TC]]/../../../../sysroot/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, soft float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -msoft-float -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-SF-32R2 %s
// CHECK-BE-UC-SF-32R2: "-internal-isystem"
// CHECK-BE-UC-SF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-UC-SF-32R2: "-internal-isystem"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc/sof"
// CHECK-BE-UC-SF-32R2: "-internal-isystem"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-UC-SF-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-SF-32R2: "[[TC]]/include"
// CHECK-BE-UC-SF-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-BE-UC-SF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-SF-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc/sof"
// CHECK-BE-UC-SF-32R2: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-SF-32R2: "[[TC]]/uclibc/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-SF-32R2: "-L[[TC]]/uclibc/sof"
// CHECK-BE-UC-SF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc/sof"
// CHECK-BE-UC-SF-32R2: "-L[[TC]]/../../../../sysroot/uclibc/sof/usr/lib/../lib"
// CHECK-BE-UC-SF-32R2: "[[TC]]/uclibc/sof{{/|\\\\}}crtend.o"
// CHECK-BE-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2 / mips16, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mips16 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-16R2 %s
// CHECK-BE-HF-16R2: "-internal-isystem"
// CHECK-BE-HF-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-16R2: "-internal-isystem"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16"
// CHECK-BE-HF-16R2: "-internal-isystem"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-16R2: "-internal-externc-isystem"
// CHECK-BE-HF-16R2: "[[TC]]/include"
// CHECK-BE-HF-16R2: "-internal-externc-isystem"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16"
// CHECK-BE-HF-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-16R2: "[[TC]]/mips16{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-16R2: "-L[[TC]]/mips16"
// CHECK-BE-HF-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16"
// CHECK-BE-HF-16R2: "-L[[TC]]/../../../../sysroot/mips16/usr/lib/../lib"
// CHECK-BE-HF-16R2: "[[TC]]/mips16{{/|\\\\}}crtend.o"
// CHECK-BE-HF-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2 / mips16, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mips16 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-16R2 %s
// CHECK-BE-HF64-16R2: "-internal-isystem"
// CHECK-BE-HF64-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-16R2: "-internal-isystem"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16"
// CHECK-BE-HF64-16R2: "-internal-isystem"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-16R2: "-internal-externc-isystem"
// CHECK-BE-HF64-16R2: "[[TC]]/include"
// CHECK-BE-HF64-16R2: "-internal-externc-isystem"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16"
// CHECK-BE-HF64-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-16R2: "[[TC]]/mips16{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-16R2: "-L[[TC]]/mips16"
// CHECK-BE-HF64-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16"
// CHECK-BE-HF64-16R2: "-L[[TC]]/../../../../sysroot/mips16/usr/lib/../lib"
// CHECK-BE-HF64-16R2: "[[TC]]/mips16{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2 / mips16, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mips16 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-16R2 %s
// CHECK-BE-SF-16R2: "-internal-isystem"
// CHECK-BE-SF-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-16R2: "-internal-isystem"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/sof"
// CHECK-BE-SF-16R2: "-internal-isystem"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-16R2: "-internal-externc-isystem"
// CHECK-BE-SF-16R2: "[[TC]]/include"
// CHECK-BE-SF-16R2: "-internal-externc-isystem"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/sof"
// CHECK-BE-SF-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../sysroot/mips16/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../sysroot/mips16/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-16R2: "[[TC]]/mips16/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-16R2: "-L[[TC]]/mips16/sof"
// CHECK-BE-SF-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/sof"
// CHECK-BE-SF-16R2: "-L[[TC]]/../../../../sysroot/mips16/sof/usr/lib/../lib"
// CHECK-BE-SF-16R2: "[[TC]]/mips16/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-16R2: "[[TC]]/../../../../sysroot/mips16/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2 / mips16, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mips16 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-16R2 %s
// CHECK-BE-NAN-16R2: "-internal-isystem"
// CHECK-BE-NAN-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-16R2: "-internal-isystem"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/nan2008"
// CHECK-BE-NAN-16R2: "-internal-isystem"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-16R2: "-internal-externc-isystem"
// CHECK-BE-NAN-16R2: "[[TC]]/include"
// CHECK-BE-NAN-16R2: "-internal-externc-isystem"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/nan2008"
// CHECK-BE-NAN-16R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-16R2: "[[TC]]/mips16/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-16R2: "-L[[TC]]/mips16/nan2008"
// CHECK-BE-NAN-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/nan2008"
// CHECK-BE-NAN-16R2: "-L[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib"
// CHECK-BE-NAN-16R2: "[[TC]]/mips16/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2 / mips16, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mips16 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-16R2 %s
// CHECK-BE-NAN64-16R2: "-internal-isystem"
// CHECK-BE-NAN64-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-16R2: "-internal-isystem"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/nan2008"
// CHECK-BE-NAN64-16R2: "-internal-isystem"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-16R2: "-internal-externc-isystem"
// CHECK-BE-NAN64-16R2: "[[TC]]/include"
// CHECK-BE-NAN64-16R2: "-internal-externc-isystem"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/nan2008"
// CHECK-BE-NAN64-16R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-16R2: "[[TC]]/mips16/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-16R2: "-L[[TC]]/mips16/nan2008"
// CHECK-BE-NAN64-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/nan2008"
// CHECK-BE-NAN64-16R2: "-L[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-16R2: "[[TC]]/mips16/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-32R2 %s
// CHECK-BE-NAN-32R2: "-internal-isystem"
// CHECK-BE-NAN-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-32R2: "-internal-isystem"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/nan2008"
// CHECK-BE-NAN-32R2: "-internal-isystem"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-32R2: "-internal-externc-isystem"
// CHECK-BE-NAN-32R2: "[[TC]]/include"
// CHECK-BE-NAN-32R2: "-internal-externc-isystem"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-32R2: "--sysroot=[[TC]]/../../../../sysroot/nan2008"
// CHECK-BE-NAN-32R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-32R2: "[[TC]]/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-32R2: "-L[[TC]]/nan2008"
// CHECK-BE-NAN-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/nan2008"
// CHECK-BE-NAN-32R2: "-L[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib"
// CHECK-BE-NAN-32R2: "[[TC]]/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, nan2008, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mnan=2008 -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-UC-NAN-32R2 %s
// CHECK-BE-UC-NAN-32R2: "-internal-isystem"
// CHECK-BE-UC-NAN-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-UC-NAN-32R2: "-internal-isystem"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc/nan2008"
// CHECK-BE-UC-NAN-32R2: "-internal-isystem"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-UC-NAN-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/include"
// CHECK-BE-UC-NAN-32R2: "-internal-externc-isystem"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-BE-UC-NAN-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-UC-NAN-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc/nan2008"
// CHECK-BE-UC-NAN-32R2: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/uclibc/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-UC-NAN-32R2: "-L[[TC]]/uclibc/nan2008"
// CHECK-BE-UC-NAN-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc/nan2008"
// CHECK-BE-UC-NAN-32R2: "-L[[TC]]/../../../../sysroot/uclibc/nan2008/usr/lib/../lib"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/uclibc/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r2, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r2 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-32R2 %s
// CHECK-BE-NAN64-32R2: "-internal-isystem"
// CHECK-BE-NAN64-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-32R2: "-internal-isystem"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/nan2008"
// CHECK-BE-NAN64-32R2: "-internal-isystem"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-32R2: "-internal-externc-isystem"
// CHECK-BE-NAN64-32R2: "[[TC]]/include"
// CHECK-BE-NAN64-32R2: "-internal-externc-isystem"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-32R2: "--sysroot=[[TC]]/../../../../sysroot/nan2008"
// CHECK-BE-NAN64-32R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-32R2: "[[TC]]/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-32R2: "-L[[TC]]/nan2008"
// CHECK-BE-NAN64-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/nan2008"
// CHECK-BE-NAN64-32R2: "-L[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-32R2: "[[TC]]/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-32R2: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, default (mips32r2), fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-32R2-DEF %s
// CHECK-BE-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-BE-NAN64-32R2-DEF: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/nan2008"
// CHECK-BE-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-32R2-DEF: "-internal-externc-isystem"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/include"
// CHECK-BE-NAN64-32R2-DEF: "-internal-externc-isystem"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-32R2-DEF: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-32R2-DEF: "--sysroot=[[TC]]/../../../../sysroot/nan2008"
// CHECK-BE-NAN64-32R2-DEF: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-32R2-DEF: "-L[[TC]]/nan2008"
// CHECK-BE-NAN64-32R2-DEF: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/nan2008"
// CHECK-BE-NAN64-32R2-DEF: "-L[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, micromips, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mmicromips -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-MM %s
// CHECK-BE-HF-MM: "-internal-isystem"
// CHECK-BE-HF-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-MM: "-internal-isystem"
// CHECK-BE-HF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips"
// CHECK-BE-HF-MM: "-internal-isystem"
// CHECK-BE-HF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-MM: "-internal-externc-isystem"
// CHECK-BE-HF-MM: "[[TC]]/include"
// CHECK-BE-HF-MM: "-internal-externc-isystem"
// CHECK-BE-HF-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips"
// CHECK-BE-HF-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-MM: "[[TC]]/micromips{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-MM: "-L[[TC]]/micromips"
// CHECK-BE-HF-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips"
// CHECK-BE-HF-MM: "-L[[TC]]/../../../../sysroot/micromips/usr/lib/../lib"
// CHECK-BE-HF-MM: "[[TC]]/micromips{{/|\\\\}}crtend.o"
// CHECK-BE-HF-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, micromips, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mmicromips -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-MM %s
// CHECK-BE-HF64-MM: "-internal-isystem"
// CHECK-BE-HF64-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-MM: "-internal-isystem"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips"
// CHECK-BE-HF64-MM: "-internal-isystem"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-MM: "-internal-externc-isystem"
// CHECK-BE-HF64-MM: "[[TC]]/include"
// CHECK-BE-HF64-MM: "-internal-externc-isystem"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips"
// CHECK-BE-HF64-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-MM: "[[TC]]/micromips{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-MM: "-L[[TC]]/micromips"
// CHECK-BE-HF64-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips"
// CHECK-BE-HF64-MM: "-L[[TC]]/../../../../sysroot/micromips/usr/lib/../lib"
// CHECK-BE-HF64-MM: "[[TC]]/micromips{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-MM: "[[TC]]/../../../../sysroot/micromips/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, micromips, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mmicromips -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-MM %s
// CHECK-BE-SF-MM: "-internal-isystem"
// CHECK-BE-SF-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-MM: "-internal-isystem"
// CHECK-BE-SF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/sof"
// CHECK-BE-SF-MM: "-internal-isystem"
// CHECK-BE-SF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-MM: "-internal-externc-isystem"
// CHECK-BE-SF-MM: "[[TC]]/include"
// CHECK-BE-SF-MM: "-internal-externc-isystem"
// CHECK-BE-SF-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/sof"
// CHECK-BE-SF-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-SF-MM: "[[TC]]/../../../../sysroot/micromips/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-MM: "[[TC]]/../../../../sysroot/micromips/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-MM: "[[TC]]/micromips/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-MM: "-L[[TC]]/micromips/sof"
// CHECK-BE-SF-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/sof"
// CHECK-BE-SF-MM: "-L[[TC]]/../../../../sysroot/micromips/sof/usr/lib/../lib"
// CHECK-BE-SF-MM: "[[TC]]/micromips/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-MM: "[[TC]]/../../../../sysroot/micromips/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, micromips, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mmicromips -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-MM %s
// CHECK-BE-NAN-MM: "-internal-isystem"
// CHECK-BE-NAN-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-MM: "-internal-isystem"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/nan2008"
// CHECK-BE-NAN-MM: "-internal-isystem"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-MM: "-internal-externc-isystem"
// CHECK-BE-NAN-MM: "[[TC]]/include"
// CHECK-BE-NAN-MM: "-internal-externc-isystem"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/nan2008"
// CHECK-BE-NAN-MM: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-MM: "[[TC]]/micromips/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-MM: "-L[[TC]]/micromips/nan2008"
// CHECK-BE-NAN-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/nan2008"
// CHECK-BE-NAN-MM: "-L[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib"
// CHECK-BE-NAN-MM: "[[TC]]/micromips/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, micromips, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mmicromips -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-MM %s
// CHECK-BE-NAN64-MM: "-internal-isystem"
// CHECK-BE-NAN64-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-MM: "-internal-isystem"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/nan2008"
// CHECK-BE-NAN64-MM: "-internal-isystem"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-MM: "-internal-externc-isystem"
// CHECK-BE-NAN64-MM: "[[TC]]/include"
// CHECK-BE-NAN64-MM: "-internal-externc-isystem"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/nan2008"
// CHECK-BE-NAN64-MM: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-MM: "[[TC]]/micromips/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-MM: "-L[[TC]]/micromips/nan2008"
// CHECK-BE-NAN64-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/nan2008"
// CHECK-BE-NAN64-MM: "-L[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib"
// CHECK-BE-NAN64-MM: "[[TC]]/micromips/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI n32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=n32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64-N32 %s
// CHECK-BE-HF-64-N32: "-internal-isystem"
// CHECK-BE-HF-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64-N32: "-internal-isystem"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64"
// CHECK-BE-HF-64-N32: "-internal-isystem"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64-N32: "-internal-externc-isystem"
// CHECK-BE-HF-64-N32: "[[TC]]/include"
// CHECK-BE-HF-64-N32: "-internal-externc-isystem"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64"
// CHECK-BE-HF-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64-N32: "[[TC]]/mips64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64-N32: "-L[[TC]]/mips64"
// CHECK-BE-HF-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64"
// CHECK-BE-HF-64-N32: "-L[[TC]]/../../../../sysroot/mips64/usr/lib"
// CHECK-BE-HF-64-N32: "[[TC]]/mips64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI n32, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=n32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-64-N32 %s
// CHECK-BE-HF64-64-N32: "-internal-isystem"
// CHECK-BE-HF64-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-64-N32: "-internal-isystem"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64"
// CHECK-BE-HF64-64-N32: "-internal-isystem"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-64-N32: "-internal-externc-isystem"
// CHECK-BE-HF64-64-N32: "[[TC]]/include"
// CHECK-BE-HF64-64-N32: "-internal-externc-isystem"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64"
// CHECK-BE-HF64-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-64-N32: "[[TC]]/mips64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-64-N32: "-L[[TC]]/mips64"
// CHECK-BE-HF64-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64"
// CHECK-BE-HF64-64-N32: "-L[[TC]]/../../../../sysroot/mips64/usr/lib"
// CHECK-BE-HF64-64-N32: "[[TC]]/mips64{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI n32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=n32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64-N32 %s
// CHECK-BE-SF-64-N32: "-internal-isystem"
// CHECK-BE-SF-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-64-N32: "-internal-isystem"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/sof"
// CHECK-BE-SF-64-N32: "-internal-isystem"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-64-N32: "-internal-externc-isystem"
// CHECK-BE-SF-64-N32: "[[TC]]/include"
// CHECK-BE-SF-64-N32: "-internal-externc-isystem"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/sof"
// CHECK-BE-SF-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-64-N32: "[[TC]]/mips64/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-64-N32: "-L[[TC]]/mips64/sof"
// CHECK-BE-SF-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/sof"
// CHECK-BE-SF-64-N32: "-L[[TC]]/../../../../sysroot/mips64/sof/usr/lib"
// CHECK-BE-SF-64-N32: "[[TC]]/mips64/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI n32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=n32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-64-N32 %s
// CHECK-BE-NAN-64-N32: "-internal-isystem"
// CHECK-BE-NAN-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-64-N32: "-internal-isystem"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/nan2008"
// CHECK-BE-NAN-64-N32: "-internal-isystem"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-64-N32: "-internal-externc-isystem"
// CHECK-BE-NAN-64-N32: "[[TC]]/include"
// CHECK-BE-NAN-64-N32: "-internal-externc-isystem"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/nan2008"
// CHECK-BE-NAN-64-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-64-N32: "[[TC]]/mips64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-64-N32: "-L[[TC]]/mips64/nan2008"
// CHECK-BE-NAN-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/nan2008"
// CHECK-BE-NAN-64-N32: "-L[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib"
// CHECK-BE-NAN-64-N32: "[[TC]]/mips64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI n32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=n32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-64-N32 %s
// CHECK-BE-NAN64-64-N32: "-internal-isystem"
// CHECK-BE-NAN64-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-64-N32: "-internal-isystem"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/nan2008"
// CHECK-BE-NAN64-64-N32: "-internal-isystem"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-64-N32: "-internal-externc-isystem"
// CHECK-BE-NAN64-64-N32: "[[TC]]/include"
// CHECK-BE-NAN64-64-N32: "-internal-externc-isystem"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/nan2008"
// CHECK-BE-NAN64-64-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-64-N32: "[[TC]]/mips64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-64-N32: "-L[[TC]]/mips64/nan2008"
// CHECK-BE-NAN64-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/nan2008"
// CHECK-BE-NAN64-64-N32: "-L[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib"
// CHECK-BE-NAN64-64-N32: "[[TC]]/mips64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64-64 %s
// CHECK-BE-HF-64-64: "-internal-isystem"
// CHECK-BE-HF-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64-64: "-internal-isystem"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64"
// CHECK-BE-HF-64-64: "-internal-isystem"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64-64: "-internal-externc-isystem"
// CHECK-BE-HF-64-64: "[[TC]]/include"
// CHECK-BE-HF-64-64: "-internal-externc-isystem"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64"
// CHECK-BE-HF-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64-64: "[[TC]]/mips64/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64-64: "-L[[TC]]/mips64/64"
// CHECK-BE-HF-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64"
// CHECK-BE-HF-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/usr/lib"
// CHECK-BE-HF-64-64: "[[TC]]/mips64/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI 64, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=64 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-64-64 %s
// CHECK-BE-HF64-64-64: "-internal-isystem"
// CHECK-BE-HF64-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-64-64: "-internal-isystem"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64"
// CHECK-BE-HF64-64-64: "-internal-isystem"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-64-64: "-internal-externc-isystem"
// CHECK-BE-HF64-64-64: "[[TC]]/include"
// CHECK-BE-HF64-64-64: "-internal-externc-isystem"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64"
// CHECK-BE-HF64-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-64-64: "[[TC]]/mips64/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-64-64: "-L[[TC]]/mips64/64"
// CHECK-BE-HF64-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64"
// CHECK-BE-HF64-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/usr/lib"
// CHECK-BE-HF64-64-64: "[[TC]]/mips64/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI 64, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=64 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64-64 %s
// CHECK-BE-SF-64-64: "-internal-isystem"
// CHECK-BE-SF-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-64-64: "-internal-isystem"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/sof"
// CHECK-BE-SF-64-64: "-internal-isystem"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-64-64: "-internal-externc-isystem"
// CHECK-BE-SF-64-64: "[[TC]]/include"
// CHECK-BE-SF-64-64: "-internal-externc-isystem"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/sof"
// CHECK-BE-SF-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-64-64: "[[TC]]/mips64/64/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-64-64: "-L[[TC]]/mips64/64/sof"
// CHECK-BE-SF-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/sof"
// CHECK-BE-SF-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/sof/usr/lib"
// CHECK-BE-SF-64-64: "[[TC]]/mips64/64/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI 64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-64-64 %s
// CHECK-BE-NAN-64-64: "-internal-isystem"
// CHECK-BE-NAN-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-64-64: "-internal-isystem"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/nan2008"
// CHECK-BE-NAN-64-64: "-internal-isystem"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-64-64: "-internal-externc-isystem"
// CHECK-BE-NAN-64-64: "[[TC]]/include"
// CHECK-BE-NAN-64-64: "-internal-externc-isystem"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/nan2008"
// CHECK-BE-NAN-64-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-64-64: "[[TC]]/mips64/64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-64-64: "-L[[TC]]/mips64/64/nan2008"
// CHECK-BE-NAN-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/nan2008"
// CHECK-BE-NAN-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib"
// CHECK-BE-NAN-64-64: "[[TC]]/mips64/64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64, ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64 -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-64-64 %s
// CHECK-BE-NAN64-64-64: "-internal-isystem"
// CHECK-BE-NAN64-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-64-64: "-internal-isystem"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/nan2008"
// CHECK-BE-NAN64-64-64: "-internal-isystem"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-64-64: "-internal-externc-isystem"
// CHECK-BE-NAN64-64-64: "[[TC]]/include"
// CHECK-BE-NAN64-64-64: "-internal-externc-isystem"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/nan2008"
// CHECK-BE-NAN64-64-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-64-64: "[[TC]]/mips64/64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-64-64: "-L[[TC]]/mips64/64/nan2008"
// CHECK-BE-NAN64-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/nan2008"
// CHECK-BE-NAN64-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib"
// CHECK-BE-NAN64-64-64: "[[TC]]/mips64/64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI n32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=n32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64R2-N32 %s
// CHECK-BE-HF-64R2-N32: "-internal-isystem"
// CHECK-BE-HF-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64R2-N32: "-internal-isystem"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2"
// CHECK-BE-HF-64R2-N32: "-internal-isystem"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-HF-64R2-N32: "[[TC]]/include"
// CHECK-BE-HF-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2"
// CHECK-BE-HF-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64R2-N32: "[[TC]]/mips64r2{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64R2-N32: "-L[[TC]]/mips64r2"
// CHECK-BE-HF-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2"
// CHECK-BE-HF-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/usr/lib"
// CHECK-BE-HF-64R2-N32: "[[TC]]/mips64r2{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI n32, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=n32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-64R2-N32 %s
// CHECK-BE-HF64-64R2-N32: "-internal-isystem"
// CHECK-BE-HF64-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-64R2-N32: "-internal-isystem"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2"
// CHECK-BE-HF64-64R2-N32: "-internal-isystem"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/include"
// CHECK-BE-HF64-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2"
// CHECK-BE-HF64-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/mips64r2{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-64R2-N32: "-L[[TC]]/mips64r2"
// CHECK-BE-HF64-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2"
// CHECK-BE-HF64-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/usr/lib"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/mips64r2{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI n32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=n32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64R2-N32 %s
// CHECK-BE-SF-64R2-N32: "-internal-isystem"
// CHECK-BE-SF-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-64R2-N32: "-internal-isystem"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/sof"
// CHECK-BE-SF-64R2-N32: "-internal-isystem"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-SF-64R2-N32: "[[TC]]/include"
// CHECK-BE-SF-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/sof"
// CHECK-BE-SF-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-64R2-N32: "[[TC]]/mips64r2/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-64R2-N32: "-L[[TC]]/mips64r2/sof"
// CHECK-BE-SF-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/sof"
// CHECK-BE-SF-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/sof/usr/lib"
// CHECK-BE-SF-64R2-N32: "[[TC]]/mips64r2/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI n32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=n32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-64R2-N32 %s
// CHECK-BE-NAN-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/nan2008"
// CHECK-BE-NAN-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/include"
// CHECK-BE-NAN-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/nan2008"
// CHECK-BE-NAN-64R2-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/mips64r2/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-64R2-N32: "-L[[TC]]/mips64r2/nan2008"
// CHECK-BE-NAN-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/nan2008"
// CHECK-BE-NAN-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/mips64r2/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI n32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=n32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-64R2-N32 %s
// CHECK-BE-NAN64-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN64-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/nan2008"
// CHECK-BE-NAN64-64R2-N32: "-internal-isystem"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/include"
// CHECK-BE-NAN64-64R2-N32: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/nan2008"
// CHECK-BE-NAN64-64R2-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/mips64r2/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-64R2-N32: "-L[[TC]]/mips64r2/nan2008"
// CHECK-BE-NAN64-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/nan2008"
// CHECK-BE-NAN64-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/mips64r2/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64R2-64 %s
// CHECK-BE-HF-64R2-64: "-internal-isystem"
// CHECK-BE-HF-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64R2-64: "-internal-isystem"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64"
// CHECK-BE-HF-64R2-64: "-internal-isystem"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64R2-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R2-64: "[[TC]]/include"
// CHECK-BE-HF-64R2-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64"
// CHECK-BE-HF-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64R2-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64R2-64: "-L[[TC]]/mips64r2/64"
// CHECK-BE-HF-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64"
// CHECK-BE-HF-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/usr/lib"
// CHECK-BE-HF-64R2-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI 64, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=64 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF64-64R2-64 %s
// CHECK-BE-HF64-64R2-64: "-internal-isystem"
// CHECK-BE-HF64-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF64-64R2-64: "-internal-isystem"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64"
// CHECK-BE-HF64-64R2-64: "-internal-isystem"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF64-64R2-64: "-internal-externc-isystem"
// CHECK-BE-HF64-64R2-64: "[[TC]]/include"
// CHECK-BE-HF64-64R2-64: "-internal-externc-isystem"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF64-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF64-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64"
// CHECK-BE-HF64-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF64-64R2-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF64-64R2-64: "-L[[TC]]/mips64r2/64"
// CHECK-BE-HF64-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64"
// CHECK-BE-HF64-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/usr/lib"
// CHECK-BE-HF64-64R2-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI 64, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=64 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64R2-64 %s
// CHECK-BE-SF-64R2-64: "-internal-isystem"
// CHECK-BE-SF-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-SF-64R2-64: "-internal-isystem"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/sof"
// CHECK-BE-SF-64R2-64: "-internal-isystem"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-SF-64R2-64: "-internal-externc-isystem"
// CHECK-BE-SF-64R2-64: "[[TC]]/include"
// CHECK-BE-SF-64R2-64: "-internal-externc-isystem"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-SF-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-SF-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/sof"
// CHECK-BE-SF-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-SF-64R2-64: "[[TC]]/mips64r2/64/sof{{/|\\\\}}crtbegin.o"
// CHECK-BE-SF-64R2-64: "-L[[TC]]/mips64r2/64/sof"
// CHECK-BE-SF-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/sof"
// CHECK-BE-SF-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/sof/usr/lib"
// CHECK-BE-SF-64R2-64: "[[TC]]/mips64r2/64/sof{{/|\\\\}}crtend.o"
// CHECK-BE-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI 64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN-64R2-64 %s
// CHECK-BE-NAN-64R2-64: "-internal-isystem"
// CHECK-BE-NAN-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN-64R2-64: "-internal-isystem"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/nan2008"
// CHECK-BE-NAN-64R2-64: "-internal-isystem"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN-64R2-64: "-internal-externc-isystem"
// CHECK-BE-NAN-64R2-64: "[[TC]]/include"
// CHECK-BE-NAN-64R2-64: "-internal-externc-isystem"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/nan2008"
// CHECK-BE-NAN-64R2-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN-64R2-64: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN-64R2-64: "-L[[TC]]/mips64r2/64/nan2008"
// CHECK-BE-NAN-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/nan2008"
// CHECK-BE-NAN-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib"
// CHECK-BE-NAN-64R2-64: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r2, ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r2 -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-64R2-64 %s
// CHECK-BE-NAN64-64R2-64: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-64R2-64: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-64R2-64: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/include"
// CHECK-BE-NAN64-64R2-64: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-64R2-64: "-L[[TC]]/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, default (mips64r2), ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-NAN64-64R2-64-DEF %s
// CHECK-BE-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-NAN64-64R2-64-DEF: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/include"
// CHECK-BE-NAN64-64R2-64-DEF: "-internal-externc-isystem"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-NAN64-64R2-64-DEF: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-NAN64-64R2-64-DEF: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64-DEF: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-BE-NAN64-64R2-64-DEF: "-L[[TC]]/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64-DEF: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/nan2008"
// CHECK-BE-NAN64-64R2-64-DEF: "-L[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/mips64r2/64/nan2008{{/|\\\\}}crtend.o"
// CHECK-BE-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-32 %s
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/el"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/include"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/el"
// CHECK-EL-HF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-32: "[[TC]]/mips32/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-32: "-L[[TC]]/mips32/el"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/el"
// CHECK-EL-HF-32: "-L[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib"
// CHECK-EL-HF-32: "[[TC]]/mips32/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-32 %s
// CHECK-EL-HF64-32: "-internal-isystem"
// CHECK-EL-HF64-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-32: "-internal-isystem"
// CHECK-EL-HF64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/el"
// CHECK-EL-HF64-32: "-internal-isystem"
// CHECK-EL-HF64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-32: "-internal-externc-isystem"
// CHECK-EL-HF64-32: "[[TC]]/include"
// CHECK-EL-HF64-32: "-internal-externc-isystem"
// CHECK-EL-HF64-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/el"
// CHECK-EL-HF64-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF64-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-32: "[[TC]]/mips32/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-32: "-L[[TC]]/mips32/el"
// CHECK-EL-HF64-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/el"
// CHECK-EL-HF64-32: "-L[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib"
// CHECK-EL-HF64-32: "[[TC]]/mips32/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-32: "[[TC]]/../../../../sysroot/mips32/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-32 %s
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/el/sof"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/include"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/el/sof"
// CHECK-EL-SF-32: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-32: "[[TC]]/../../../../sysroot/mips32/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../sysroot/mips32/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-32: "[[TC]]/mips32/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-32: "-L[[TC]]/mips32/el/sof"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/el/sof"
// CHECK-EL-SF-32: "-L[[TC]]/../../../../sysroot/mips32/el/sof/usr/lib/../lib"
// CHECK-EL-SF-32: "[[TC]]/mips32/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-32: "[[TC]]/../../../../sysroot/mips32/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32 / mips16, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mips16 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-16 %s
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/el"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/include"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/el"
// CHECK-EL-HF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-16: "[[TC]]/mips32/mips16/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-16: "-L[[TC]]/mips32/mips16/el"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/el"
// CHECK-EL-HF-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib"
// CHECK-EL-HF-16: "[[TC]]/mips32/mips16/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32 / mips16, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mips16 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-16 %s
// CHECK-EL-HF64-16: "-internal-isystem"
// CHECK-EL-HF64-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-16: "-internal-isystem"
// CHECK-EL-HF64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/el"
// CHECK-EL-HF64-16: "-internal-isystem"
// CHECK-EL-HF64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-16: "-internal-externc-isystem"
// CHECK-EL-HF64-16: "[[TC]]/include"
// CHECK-EL-HF64-16: "-internal-externc-isystem"
// CHECK-EL-HF64-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/el"
// CHECK-EL-HF64-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-16: "[[TC]]/mips32/mips16/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-16: "-L[[TC]]/mips32/mips16/el"
// CHECK-EL-HF64-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/el"
// CHECK-EL-HF64-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib"
// CHECK-EL-HF64-16: "[[TC]]/mips32/mips16/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32 / mips16, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mips16 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-16 %s
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/el/sof"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/include"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/el/sof"
// CHECK-EL-SF-16: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-16: "[[TC]]/mips32/mips16/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-16: "-L[[TC]]/mips32/mips16/el/sof"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/el/sof"
// CHECK-EL-SF-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/el/sof/usr/lib/../lib"
// CHECK-EL-SF-16: "[[TC]]/mips32/mips16/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32 / mips16, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mips16 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-16 %s
// CHECK-EL-NAN-16: "-internal-isystem"
// CHECK-EL-NAN-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-16: "-internal-isystem"
// CHECK-EL-NAN-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/el/nan2008"
// CHECK-EL-NAN-16: "-internal-isystem"
// CHECK-EL-NAN-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-16: "-internal-externc-isystem"
// CHECK-EL-NAN-16: "[[TC]]/include"
// CHECK-EL-NAN-16: "-internal-externc-isystem"
// CHECK-EL-NAN-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008"
// CHECK-EL-NAN-16: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-16: "[[TC]]/mips32/mips16/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-16: "-L[[TC]]/mips32/mips16/el/nan2008"
// CHECK-EL-NAN-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/el/nan2008"
// CHECK-EL-NAN-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN-16: "[[TC]]/mips32/mips16/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32 / mips16, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mips16 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-16 %s
// CHECK-EL-NAN64-16: "-internal-isystem"
// CHECK-EL-NAN64-16: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-16: "-internal-isystem"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/mips16/el/nan2008"
// CHECK-EL-NAN64-16: "-internal-isystem"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-16: "-internal-externc-isystem"
// CHECK-EL-NAN64-16: "[[TC]]/include"
// CHECK-EL-NAN64-16: "-internal-externc-isystem"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-16: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-16: "--sysroot=[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008"
// CHECK-EL-NAN64-16: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-16: "[[TC]]/mips32/mips16/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-16: "-L[[TC]]/mips32/mips16/el/nan2008"
// CHECK-EL-NAN64-16: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/mips16/el/nan2008"
// CHECK-EL-NAN64-16: "-L[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-16: "[[TC]]/mips32/mips16/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-16: "[[TC]]/../../../../sysroot/mips32/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-32 %s
// CHECK-EL-NAN-32: "-internal-isystem"
// CHECK-EL-NAN-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-32: "-internal-isystem"
// CHECK-EL-NAN-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/el/nan2008"
// CHECK-EL-NAN-32: "-internal-isystem"
// CHECK-EL-NAN-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-32: "-internal-externc-isystem"
// CHECK-EL-NAN-32: "[[TC]]/include"
// CHECK-EL-NAN-32: "-internal-externc-isystem"
// CHECK-EL-NAN-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/el/nan2008"
// CHECK-EL-NAN-32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-32: "[[TC]]/mips32/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-32: "-L[[TC]]/mips32/el/nan2008"
// CHECK-EL-NAN-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/el/nan2008"
// CHECK-EL-NAN-32: "-L[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN-32: "[[TC]]/mips32/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-32 %s
// CHECK-EL-NAN64-32: "-internal-isystem"
// CHECK-EL-NAN64-32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-32: "-internal-isystem"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips32/el/nan2008"
// CHECK-EL-NAN64-32: "-internal-isystem"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-32: "-internal-externc-isystem"
// CHECK-EL-NAN64-32: "[[TC]]/include"
// CHECK-EL-NAN64-32: "-internal-externc-isystem"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-32: "--sysroot=[[TC]]/../../../../sysroot/mips32/el/nan2008"
// CHECK-EL-NAN64-32: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-32: "[[TC]]/mips32/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-32: "-L[[TC]]/mips32/el/nan2008"
// CHECK-EL-NAN64-32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips32/el/nan2008"
// CHECK-EL-NAN64-32: "-L[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-32: "[[TC]]/mips32/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-32: "[[TC]]/../../../../sysroot/mips32/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-32R2 %s
// CHECK-EL-HF-32R2: "-internal-isystem"
// CHECK-EL-HF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-32R2: "-internal-isystem"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el"
// CHECK-EL-HF-32R2: "-internal-isystem"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-32R2: "-internal-externc-isystem"
// CHECK-EL-HF-32R2: "[[TC]]/include"
// CHECK-EL-HF-32R2: "-internal-externc-isystem"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-32R2: "--sysroot=[[TC]]/../../../../sysroot/el"
// CHECK-EL-HF-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-32R2: "[[TC]]/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-32R2: "-L[[TC]]/el"
// CHECK-EL-HF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el"
// CHECK-EL-HF-32R2: "-L[[TC]]/../../../../sysroot/el/usr/lib/../lib"
// CHECK-EL-HF-32R2: "[[TC]]/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, hard float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mhard-float -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-HF-32R2 %s
// CHECK-EL-UC-HF-32R2: "-internal-isystem"
// CHECK-EL-UC-HF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-UC-HF-32R2: "-internal-isystem"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc/el"
// CHECK-EL-UC-HF-32R2: "-internal-isystem"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-UC-HF-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-HF-32R2: "[[TC]]/include"
// CHECK-EL-UC-HF-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-EL-UC-HF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-HF-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc/el"
// CHECK-EL-UC-HF-32R2: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-HF-32R2: "[[TC]]/uclibc/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-HF-32R2: "-L[[TC]]/uclibc/el"
// CHECK-EL-UC-HF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc/el"
// CHECK-EL-UC-HF-32R2: "-L[[TC]]/../../../../sysroot/uclibc/el/usr/lib/../lib"
// CHECK-EL-UC-HF-32R2: "[[TC]]/uclibc/el{{/|\\\\}}crtend.o"
// CHECK-EL-UC-HF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-32R2 %s
// CHECK-EL-HF64-32R2: "-internal-isystem"
// CHECK-EL-HF64-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-32R2: "-internal-isystem"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el"
// CHECK-EL-HF64-32R2: "-internal-isystem"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-32R2: "-internal-externc-isystem"
// CHECK-EL-HF64-32R2: "[[TC]]/include"
// CHECK-EL-HF64-32R2: "-internal-externc-isystem"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-32R2: "--sysroot=[[TC]]/../../../../sysroot/el"
// CHECK-EL-HF64-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-32R2: "[[TC]]/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-32R2: "-L[[TC]]/el"
// CHECK-EL-HF64-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el"
// CHECK-EL-HF64-32R2: "-L[[TC]]/../../../../sysroot/el/usr/lib/../lib"
// CHECK-EL-HF64-32R2: "[[TC]]/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-32R2: "[[TC]]/../../../../sysroot/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-32R2 %s
// CHECK-EL-SF-32R2: "-internal-isystem"
// CHECK-EL-SF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-32R2: "-internal-isystem"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el/sof"
// CHECK-EL-SF-32R2: "-internal-isystem"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-32R2: "-internal-externc-isystem"
// CHECK-EL-SF-32R2: "[[TC]]/include"
// CHECK-EL-SF-32R2: "-internal-externc-isystem"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-32R2: "--sysroot=[[TC]]/../../../../sysroot/el/sof"
// CHECK-EL-SF-32R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../sysroot/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../sysroot/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-32R2: "[[TC]]/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-32R2: "-L[[TC]]/el/sof"
// CHECK-EL-SF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el/sof"
// CHECK-EL-SF-32R2: "-L[[TC]]/../../../../sysroot/el/sof/usr/lib/../lib"
// CHECK-EL-SF-32R2: "[[TC]]/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-32R2: "[[TC]]/../../../../sysroot/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, soft float, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -msoft-float -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-SF-32R2 %s
// CHECK-EL-UC-SF-32R2: "-internal-isystem"
// CHECK-EL-UC-SF-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-UC-SF-32R2: "-internal-isystem"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc/el/sof"
// CHECK-EL-UC-SF-32R2: "-internal-isystem"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-UC-SF-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-SF-32R2: "[[TC]]/include"
// CHECK-EL-UC-SF-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-EL-UC-SF-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-SF-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc/el/sof"
// CHECK-EL-UC-SF-32R2: "-dynamic-linker" "/lib/ld-uClibc.so.0"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-SF-32R2: "[[TC]]/uclibc/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-SF-32R2: "-L[[TC]]/uclibc/el/sof"
// CHECK-EL-UC-SF-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc/el/sof"
// CHECK-EL-UC-SF-32R2: "-L[[TC]]/../../../../sysroot/uclibc/el/sof/usr/lib/../lib"
// CHECK-EL-UC-SF-32R2: "[[TC]]/uclibc/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-UC-SF-32R2: "[[TC]]/../../../../sysroot/uclibc/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2 / mips16, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mips16 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-16R2 %s
// CHECK-EL-HF-16R2: "-internal-isystem"
// CHECK-EL-HF-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-16R2: "-internal-isystem"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/el"
// CHECK-EL-HF-16R2: "-internal-isystem"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-16R2: "-internal-externc-isystem"
// CHECK-EL-HF-16R2: "[[TC]]/include"
// CHECK-EL-HF-16R2: "-internal-externc-isystem"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/el"
// CHECK-EL-HF-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-16R2: "[[TC]]/mips16/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-16R2: "-L[[TC]]/mips16/el"
// CHECK-EL-HF-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/el"
// CHECK-EL-HF-16R2: "-L[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib"
// CHECK-EL-HF-16R2: "[[TC]]/mips16/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2 / mips16, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mips16 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-16R2 %s
// CHECK-EL-HF64-16R2: "-internal-isystem"
// CHECK-EL-HF64-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-16R2: "-internal-isystem"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/el"
// CHECK-EL-HF64-16R2: "-internal-isystem"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-16R2: "-internal-externc-isystem"
// CHECK-EL-HF64-16R2: "[[TC]]/include"
// CHECK-EL-HF64-16R2: "-internal-externc-isystem"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/el"
// CHECK-EL-HF64-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-16R2: "[[TC]]/mips16/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-16R2: "-L[[TC]]/mips16/el"
// CHECK-EL-HF64-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/el"
// CHECK-EL-HF64-16R2: "-L[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib"
// CHECK-EL-HF64-16R2: "[[TC]]/mips16/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-16R2: "[[TC]]/../../../../sysroot/mips16/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2 / mips16, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mips16 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-16R2 %s
// CHECK-EL-SF-16R2: "-internal-isystem"
// CHECK-EL-SF-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-16R2: "-internal-isystem"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/el/sof"
// CHECK-EL-SF-16R2: "-internal-isystem"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-16R2: "-internal-externc-isystem"
// CHECK-EL-SF-16R2: "[[TC]]/include"
// CHECK-EL-SF-16R2: "-internal-externc-isystem"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/el/sof"
// CHECK-EL-SF-16R2: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../sysroot/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../sysroot/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-16R2: "[[TC]]/mips16/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-16R2: "-L[[TC]]/mips16/el/sof"
// CHECK-EL-SF-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/el/sof"
// CHECK-EL-SF-16R2: "-L[[TC]]/../../../../sysroot/mips16/el/sof/usr/lib/../lib"
// CHECK-EL-SF-16R2: "[[TC]]/mips16/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-16R2: "[[TC]]/../../../../sysroot/mips16/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2 / mips16, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mips16 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-16R2 %s
// CHECK-EL-NAN-16R2: "-internal-isystem"
// CHECK-EL-NAN-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-16R2: "-internal-isystem"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/el/nan2008"
// CHECK-EL-NAN-16R2: "-internal-isystem"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-16R2: "-internal-externc-isystem"
// CHECK-EL-NAN-16R2: "[[TC]]/include"
// CHECK-EL-NAN-16R2: "-internal-externc-isystem"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/el/nan2008"
// CHECK-EL-NAN-16R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-16R2: "[[TC]]/mips16/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-16R2: "-L[[TC]]/mips16/el/nan2008"
// CHECK-EL-NAN-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/el/nan2008"
// CHECK-EL-NAN-16R2: "-L[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN-16R2: "[[TC]]/mips16/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2 / mips16, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mips16 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-16R2 %s
// CHECK-EL-NAN64-16R2: "-internal-isystem"
// CHECK-EL-NAN64-16R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-16R2: "-internal-isystem"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips16/el/nan2008"
// CHECK-EL-NAN64-16R2: "-internal-isystem"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-16R2: "-internal-externc-isystem"
// CHECK-EL-NAN64-16R2: "[[TC]]/include"
// CHECK-EL-NAN64-16R2: "-internal-externc-isystem"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-16R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-16R2: "--sysroot=[[TC]]/../../../../sysroot/mips16/el/nan2008"
// CHECK-EL-NAN64-16R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-16R2: "[[TC]]/mips16/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-16R2: "-L[[TC]]/mips16/el/nan2008"
// CHECK-EL-NAN64-16R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/mips16/el/nan2008"
// CHECK-EL-NAN64-16R2: "-L[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-16R2: "[[TC]]/mips16/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-16R2: "[[TC]]/../../../../sysroot/mips16/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-32R2 %s
// CHECK-EL-NAN-32R2: "-internal-isystem"
// CHECK-EL-NAN-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-32R2: "-internal-isystem"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el/nan2008"
// CHECK-EL-NAN-32R2: "-internal-isystem"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-32R2: "-internal-externc-isystem"
// CHECK-EL-NAN-32R2: "[[TC]]/include"
// CHECK-EL-NAN-32R2: "-internal-externc-isystem"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-32R2: "--sysroot=[[TC]]/../../../../sysroot/el/nan2008"
// CHECK-EL-NAN-32R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-32R2: "[[TC]]/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-32R2: "-L[[TC]]/el/nan2008"
// CHECK-EL-NAN-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el/nan2008"
// CHECK-EL-NAN-32R2: "-L[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN-32R2: "[[TC]]/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, nan2008, uclibc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mnan=2008 -muclibc \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-UC-NAN-32R2 %s
// CHECK-EL-UC-NAN-32R2: "-internal-isystem"
// CHECK-EL-UC-NAN-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-UC-NAN-32R2: "-internal-isystem"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/uclibc/el/nan2008"
// CHECK-EL-UC-NAN-32R2: "-internal-isystem"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-UC-NAN-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/include"
// CHECK-EL-UC-NAN-32R2: "-internal-externc-isystem"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/usr/include"
// CHECK-EL-UC-NAN-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-UC-NAN-32R2: "--sysroot=[[TC]]/../../../../sysroot/uclibc/el/nan2008"
// CHECK-EL-UC-NAN-32R2: "-dynamic-linker" "/lib/ld-uClibc-mipsn8.so.0"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/uclibc/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-UC-NAN-32R2: "-L[[TC]]/uclibc/el/nan2008"
// CHECK-EL-UC-NAN-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/uclibc/el/nan2008"
// CHECK-EL-UC-NAN-32R2: "-L[[TC]]/../../../../sysroot/uclibc/el/nan2008/usr/lib/../lib"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/uclibc/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-UC-NAN-32R2: "[[TC]]/../../../../sysroot/uclibc/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips32r2, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mips32r2 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-32R2 %s
// CHECK-EL-NAN64-32R2: "-internal-isystem"
// CHECK-EL-NAN64-32R2: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-32R2: "-internal-isystem"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el/nan2008"
// CHECK-EL-NAN64-32R2: "-internal-isystem"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-32R2: "-internal-externc-isystem"
// CHECK-EL-NAN64-32R2: "[[TC]]/include"
// CHECK-EL-NAN64-32R2: "-internal-externc-isystem"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-32R2: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-32R2: "--sysroot=[[TC]]/../../../../sysroot/el/nan2008"
// CHECK-EL-NAN64-32R2: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-32R2: "[[TC]]/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-32R2: "-L[[TC]]/el/nan2008"
// CHECK-EL-NAN64-32R2: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el/nan2008"
// CHECK-EL-NAN64-32R2: "-L[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-32R2: "[[TC]]/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-32R2: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, default (mips32r2), fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-32R2-DEF %s
// CHECK-EL-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-EL-NAN64-32R2-DEF: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/el/nan2008"
// CHECK-EL-NAN64-32R2-DEF: "-internal-isystem"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-32R2-DEF: "-internal-externc-isystem"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/include"
// CHECK-EL-NAN64-32R2-DEF: "-internal-externc-isystem"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-32R2-DEF: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-32R2-DEF: "--sysroot=[[TC]]/../../../../sysroot/el/nan2008"
// CHECK-EL-NAN64-32R2-DEF: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-32R2-DEF: "-L[[TC]]/el/nan2008"
// CHECK-EL-NAN64-32R2-DEF: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/el/nan2008"
// CHECK-EL-NAN64-32R2-DEF: "-L[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-32R2-DEF: "[[TC]]/../../../../sysroot/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, micromips, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mmicromips -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-MM %s
// CHECK-EL-HF-MM: "-internal-isystem"
// CHECK-EL-HF-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-MM: "-internal-isystem"
// CHECK-EL-HF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/el"
// CHECK-EL-HF-MM: "-internal-isystem"
// CHECK-EL-HF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-MM: "-internal-externc-isystem"
// CHECK-EL-HF-MM: "[[TC]]/include"
// CHECK-EL-HF-MM: "-internal-externc-isystem"
// CHECK-EL-HF-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/el"
// CHECK-EL-HF-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-MM: "[[TC]]/micromips/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-MM: "-L[[TC]]/micromips/el"
// CHECK-EL-HF-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/el"
// CHECK-EL-HF-MM: "-L[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib"
// CHECK-EL-HF-MM: "[[TC]]/micromips/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, micromips, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mmicromips -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-MM %s
// CHECK-EL-HF64-MM: "-internal-isystem"
// CHECK-EL-HF64-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-MM: "-internal-isystem"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/el"
// CHECK-EL-HF64-MM: "-internal-isystem"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-MM: "-internal-externc-isystem"
// CHECK-EL-HF64-MM: "[[TC]]/include"
// CHECK-EL-HF64-MM: "-internal-externc-isystem"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/el"
// CHECK-EL-HF64-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-MM: "[[TC]]/micromips/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-MM: "-L[[TC]]/micromips/el"
// CHECK-EL-HF64-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/el"
// CHECK-EL-HF64-MM: "-L[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib"
// CHECK-EL-HF64-MM: "[[TC]]/micromips/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-MM: "[[TC]]/../../../../sysroot/micromips/el/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, micromips, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mmicromips -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-MM %s
// CHECK-EL-SF-MM: "-internal-isystem"
// CHECK-EL-SF-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-MM: "-internal-isystem"
// CHECK-EL-SF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/el/sof"
// CHECK-EL-SF-MM: "-internal-isystem"
// CHECK-EL-SF-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-MM: "-internal-externc-isystem"
// CHECK-EL-SF-MM: "[[TC]]/include"
// CHECK-EL-SF-MM: "-internal-externc-isystem"
// CHECK-EL-SF-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/el/sof"
// CHECK-EL-SF-MM: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-EL-SF-MM: "[[TC]]/../../../../sysroot/micromips/el/sof/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-MM: "[[TC]]/../../../../sysroot/micromips/el/sof/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-MM: "[[TC]]/micromips/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-MM: "-L[[TC]]/micromips/el/sof"
// CHECK-EL-SF-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/el/sof"
// CHECK-EL-SF-MM: "-L[[TC]]/../../../../sysroot/micromips/el/sof/usr/lib/../lib"
// CHECK-EL-SF-MM: "[[TC]]/micromips/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-MM: "[[TC]]/../../../../sysroot/micromips/el/sof/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, micromips, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mmicromips -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-MM %s
// CHECK-EL-NAN-MM: "-internal-isystem"
// CHECK-EL-NAN-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-MM: "-internal-isystem"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/el/nan2008"
// CHECK-EL-NAN-MM: "-internal-isystem"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-MM: "-internal-externc-isystem"
// CHECK-EL-NAN-MM: "[[TC]]/include"
// CHECK-EL-NAN-MM: "-internal-externc-isystem"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/el/nan2008"
// CHECK-EL-NAN-MM: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-MM: "[[TC]]/micromips/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-MM: "-L[[TC]]/micromips/el/nan2008"
// CHECK-EL-NAN-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/el/nan2008"
// CHECK-EL-NAN-MM: "-L[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN-MM: "[[TC]]/micromips/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, micromips, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-mti-linux-gnu -mmicromips -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-MM %s
// CHECK-EL-NAN64-MM: "-internal-isystem"
// CHECK-EL-NAN64-MM: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-MM: "-internal-isystem"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/micromips/el/nan2008"
// CHECK-EL-NAN64-MM: "-internal-isystem"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-MM: "-internal-externc-isystem"
// CHECK-EL-NAN64-MM: "[[TC]]/include"
// CHECK-EL-NAN64-MM: "-internal-externc-isystem"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-MM: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-MM: "--sysroot=[[TC]]/../../../../sysroot/micromips/el/nan2008"
// CHECK-EL-NAN64-MM: "-dynamic-linker" "/lib/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-MM: "[[TC]]/micromips/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-MM: "-L[[TC]]/micromips/el/nan2008"
// CHECK-EL-NAN64-MM: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib/micromips/el/nan2008"
// CHECK-EL-NAN64-MM: "-L[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib"
// CHECK-EL-NAN64-MM: "[[TC]]/micromips/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-MM: "[[TC]]/../../../../sysroot/micromips/el/nan2008/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI n32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=n32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64-N32 %s
// CHECK-EL-HF-64-N32: "-internal-isystem"
// CHECK-EL-HF-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-64-N32: "-internal-isystem"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/el"
// CHECK-EL-HF-64-N32: "-internal-isystem"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-64-N32: "-internal-externc-isystem"
// CHECK-EL-HF-64-N32: "[[TC]]/include"
// CHECK-EL-HF-64-N32: "-internal-externc-isystem"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/el"
// CHECK-EL-HF-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-64-N32: "[[TC]]/mips64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-64-N32: "-L[[TC]]/mips64/el"
// CHECK-EL-HF-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/el"
// CHECK-EL-HF-64-N32: "-L[[TC]]/../../../../sysroot/mips64/el/usr/lib"
// CHECK-EL-HF-64-N32: "[[TC]]/mips64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI n32, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=n32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-64-N32 %s
// CHECK-EL-HF64-64-N32: "-internal-isystem"
// CHECK-EL-HF64-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-64-N32: "-internal-isystem"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/el"
// CHECK-EL-HF64-64-N32: "-internal-isystem"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-64-N32: "-internal-externc-isystem"
// CHECK-EL-HF64-64-N32: "[[TC]]/include"
// CHECK-EL-HF64-64-N32: "-internal-externc-isystem"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/el"
// CHECK-EL-HF64-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-64-N32: "[[TC]]/mips64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-64-N32: "-L[[TC]]/mips64/el"
// CHECK-EL-HF64-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/el"
// CHECK-EL-HF64-64-N32: "-L[[TC]]/../../../../sysroot/mips64/el/usr/lib"
// CHECK-EL-HF64-64-N32: "[[TC]]/mips64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI n32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=n32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64-N32 %s
// CHECK-EL-SF-64-N32: "-internal-isystem"
// CHECK-EL-SF-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-64-N32: "-internal-isystem"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/el/sof"
// CHECK-EL-SF-64-N32: "-internal-isystem"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-64-N32: "-internal-externc-isystem"
// CHECK-EL-SF-64-N32: "[[TC]]/include"
// CHECK-EL-SF-64-N32: "-internal-externc-isystem"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/el/sof"
// CHECK-EL-SF-64-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-64-N32: "[[TC]]/mips64/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-64-N32: "-L[[TC]]/mips64/el/sof"
// CHECK-EL-SF-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/el/sof"
// CHECK-EL-SF-64-N32: "-L[[TC]]/../../../../sysroot/mips64/el/sof/usr/lib"
// CHECK-EL-SF-64-N32: "[[TC]]/mips64/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-64-N32: "[[TC]]/../../../../sysroot/mips64/el/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI n32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=n32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-64-N32 %s
// CHECK-EL-NAN-64-N32: "-internal-isystem"
// CHECK-EL-NAN-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-64-N32: "-internal-isystem"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/el/nan2008"
// CHECK-EL-NAN-64-N32: "-internal-isystem"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-64-N32: "-internal-externc-isystem"
// CHECK-EL-NAN-64-N32: "[[TC]]/include"
// CHECK-EL-NAN-64-N32: "-internal-externc-isystem"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/el/nan2008"
// CHECK-EL-NAN-64-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-64-N32: "[[TC]]/mips64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-64-N32: "-L[[TC]]/mips64/el/nan2008"
// CHECK-EL-NAN-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/el/nan2008"
// CHECK-EL-NAN-64-N32: "-L[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib"
// CHECK-EL-NAN-64-N32: "[[TC]]/mips64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI n32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=n32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-64-N32 %s
// CHECK-EL-NAN64-64-N32: "-internal-isystem"
// CHECK-EL-NAN64-64-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-64-N32: "-internal-isystem"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/el/nan2008"
// CHECK-EL-NAN64-64-N32: "-internal-isystem"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-64-N32: "-internal-externc-isystem"
// CHECK-EL-NAN64-64-N32: "[[TC]]/include"
// CHECK-EL-NAN64-64-N32: "-internal-externc-isystem"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-64-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64/el/nan2008"
// CHECK-EL-NAN64-64-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-64-N32: "[[TC]]/mips64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-64-N32: "-L[[TC]]/mips64/el/nan2008"
// CHECK-EL-NAN64-64-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/el/nan2008"
// CHECK-EL-NAN64-64-N32: "-L[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib"
// CHECK-EL-NAN64-64-N32: "[[TC]]/mips64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-64-N32: "[[TC]]/../../../../sysroot/mips64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64-64 %s
// CHECK-EL-HF-64-64: "-internal-isystem"
// CHECK-EL-HF-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-64-64: "-internal-isystem"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/el"
// CHECK-EL-HF-64-64: "-internal-isystem"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-64-64: "-internal-externc-isystem"
// CHECK-EL-HF-64-64: "[[TC]]/include"
// CHECK-EL-HF-64-64: "-internal-externc-isystem"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/el"
// CHECK-EL-HF-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-64-64: "[[TC]]/mips64/64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-64-64: "-L[[TC]]/mips64/64/el"
// CHECK-EL-HF-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/el"
// CHECK-EL-HF-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/el/usr/lib"
// CHECK-EL-HF-64-64: "[[TC]]/mips64/64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI 64, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=64 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-64-64 %s
// CHECK-EL-HF64-64-64: "-internal-isystem"
// CHECK-EL-HF64-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-64-64: "-internal-isystem"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/el"
// CHECK-EL-HF64-64-64: "-internal-isystem"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-64-64: "-internal-externc-isystem"
// CHECK-EL-HF64-64-64: "[[TC]]/include"
// CHECK-EL-HF64-64-64: "-internal-externc-isystem"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/el"
// CHECK-EL-HF64-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-64-64: "[[TC]]/mips64/64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-64-64: "-L[[TC]]/mips64/64/el"
// CHECK-EL-HF64-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/el"
// CHECK-EL-HF64-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/el/usr/lib"
// CHECK-EL-HF64-64-64: "[[TC]]/mips64/64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI 64, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=64 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64-64 %s
// CHECK-EL-SF-64-64: "-internal-isystem"
// CHECK-EL-SF-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-64-64: "-internal-isystem"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/el/sof"
// CHECK-EL-SF-64-64: "-internal-isystem"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-64-64: "-internal-externc-isystem"
// CHECK-EL-SF-64-64: "[[TC]]/include"
// CHECK-EL-SF-64-64: "-internal-externc-isystem"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/el/sof"
// CHECK-EL-SF-64-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-64-64: "[[TC]]/mips64/64/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-64-64: "-L[[TC]]/mips64/64/el/sof"
// CHECK-EL-SF-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/el/sof"
// CHECK-EL-SF-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/el/sof/usr/lib"
// CHECK-EL-SF-64-64: "[[TC]]/mips64/64/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI 64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-64-64 %s
// CHECK-EL-NAN-64-64: "-internal-isystem"
// CHECK-EL-NAN-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-64-64: "-internal-isystem"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/el/nan2008"
// CHECK-EL-NAN-64-64: "-internal-isystem"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-64-64: "-internal-externc-isystem"
// CHECK-EL-NAN-64-64: "[[TC]]/include"
// CHECK-EL-NAN-64-64: "-internal-externc-isystem"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/el/nan2008"
// CHECK-EL-NAN-64-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-64-64: "[[TC]]/mips64/64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-64-64: "-L[[TC]]/mips64/64/el/nan2008"
// CHECK-EL-NAN-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/el/nan2008"
// CHECK-EL-NAN-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib"
// CHECK-EL-NAN-64-64: "[[TC]]/mips64/64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64, ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64 -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-64-64 %s
// CHECK-EL-NAN64-64-64: "-internal-isystem"
// CHECK-EL-NAN64-64-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-64-64: "-internal-isystem"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64/64/el/nan2008"
// CHECK-EL-NAN64-64-64: "-internal-isystem"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-64-64: "-internal-externc-isystem"
// CHECK-EL-NAN64-64-64: "[[TC]]/include"
// CHECK-EL-NAN64-64-64: "-internal-externc-isystem"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-64-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-64-64: "--sysroot=[[TC]]/../../../../sysroot/mips64/64/el/nan2008"
// CHECK-EL-NAN64-64-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-64-64: "[[TC]]/mips64/64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-64-64: "-L[[TC]]/mips64/64/el/nan2008"
// CHECK-EL-NAN64-64-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64/64/el/nan2008"
// CHECK-EL-NAN64-64-64: "-L[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib"
// CHECK-EL-NAN64-64-64: "[[TC]]/mips64/64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-64-64: "[[TC]]/../../../../sysroot/mips64/64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI n32, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=n32 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64R2-N32 %s
// CHECK-EL-HF-64R2-N32: "-internal-isystem"
// CHECK-EL-HF-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-64R2-N32: "-internal-isystem"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/el"
// CHECK-EL-HF-64R2-N32: "-internal-isystem"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-HF-64R2-N32: "[[TC]]/include"
// CHECK-EL-HF-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/el"
// CHECK-EL-HF-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-64R2-N32: "[[TC]]/mips64r2/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-64R2-N32: "-L[[TC]]/mips64r2/el"
// CHECK-EL-HF-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/el"
// CHECK-EL-HF-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/el/usr/lib"
// CHECK-EL-HF-64R2-N32: "[[TC]]/mips64r2/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI n32, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=n32 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-64R2-N32 %s
// CHECK-EL-HF64-64R2-N32: "-internal-isystem"
// CHECK-EL-HF64-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-64R2-N32: "-internal-isystem"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/el"
// CHECK-EL-HF64-64R2-N32: "-internal-isystem"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/include"
// CHECK-EL-HF64-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/el"
// CHECK-EL-HF64-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/mips64r2/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-64R2-N32: "-L[[TC]]/mips64r2/el"
// CHECK-EL-HF64-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/el"
// CHECK-EL-HF64-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/el/usr/lib"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/mips64r2/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI n32, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=n32 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64R2-N32 %s
// CHECK-EL-SF-64R2-N32: "-internal-isystem"
// CHECK-EL-SF-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-64R2-N32: "-internal-isystem"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/el/sof"
// CHECK-EL-SF-64R2-N32: "-internal-isystem"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-SF-64R2-N32: "[[TC]]/include"
// CHECK-EL-SF-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/el/sof"
// CHECK-EL-SF-64R2-N32: "-dynamic-linker" "/lib32/ld.so.1"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-64R2-N32: "[[TC]]/mips64r2/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-64R2-N32: "-L[[TC]]/mips64r2/el/sof"
// CHECK-EL-SF-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/el/sof"
// CHECK-EL-SF-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/el/sof/usr/lib"
// CHECK-EL-SF-64R2-N32: "[[TC]]/mips64r2/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI n32, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=n32 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-64R2-N32 %s
// CHECK-EL-NAN-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/el/nan2008"
// CHECK-EL-NAN-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/include"
// CHECK-EL-NAN-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/el/nan2008"
// CHECK-EL-NAN-64R2-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/mips64r2/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-64R2-N32: "-L[[TC]]/mips64r2/el/nan2008"
// CHECK-EL-NAN-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/el/nan2008"
// CHECK-EL-NAN-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/mips64r2/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI n32, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=n32 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-64R2-N32 %s
// CHECK-EL-NAN64-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN64-64R2-N32: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/el/nan2008"
// CHECK-EL-NAN64-64R2-N32: "-internal-isystem"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/include"
// CHECK-EL-NAN64-64R2-N32: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-64R2-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-64R2-N32: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/el/nan2008"
// CHECK-EL-NAN64-64R2-N32: "-dynamic-linker" "/lib32/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/mips64r2/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-64R2-N32: "-L[[TC]]/mips64r2/el/nan2008"
// CHECK-EL-NAN64-64R2-N32: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/el/nan2008"
// CHECK-EL-NAN64-64R2-N32: "-L[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/mips64r2/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-64R2-N32: "[[TC]]/../../../../sysroot/mips64r2/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64R2-64 %s
// CHECK-EL-HF-64R2-64: "-internal-isystem"
// CHECK-EL-HF-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF-64R2-64: "-internal-isystem"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el"
// CHECK-EL-HF-64R2-64: "-internal-isystem"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF-64R2-64: "-internal-externc-isystem"
// CHECK-EL-HF-64R2-64: "[[TC]]/include"
// CHECK-EL-HF-64R2-64: "-internal-externc-isystem"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el"
// CHECK-EL-HF-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF-64R2-64: "[[TC]]/mips64r2/64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF-64R2-64: "-L[[TC]]/mips64r2/64/el"
// CHECK-EL-HF-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el"
// CHECK-EL-HF-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib"
// CHECK-EL-HF-64R2-64: "[[TC]]/mips64r2/64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI 64, fp64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=64 -mfp64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF64-64R2-64 %s
// CHECK-EL-HF64-64R2-64: "-internal-isystem"
// CHECK-EL-HF64-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-HF64-64R2-64: "-internal-isystem"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el"
// CHECK-EL-HF64-64R2-64: "-internal-isystem"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-HF64-64R2-64: "-internal-externc-isystem"
// CHECK-EL-HF64-64R2-64: "[[TC]]/include"
// CHECK-EL-HF64-64R2-64: "-internal-externc-isystem"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-HF64-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-HF64-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el"
// CHECK-EL-HF64-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-HF64-64R2-64: "[[TC]]/mips64r2/64/el{{/|\\\\}}crtbegin.o"
// CHECK-EL-HF64-64R2-64: "-L[[TC]]/mips64r2/64/el"
// CHECK-EL-HF64-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el"
// CHECK-EL-HF64-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib"
// CHECK-EL-HF64-64R2-64: "[[TC]]/mips64r2/64/el{{/|\\\\}}crtend.o"
// CHECK-EL-HF64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI 64, soft float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=64 -msoft-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64R2-64 %s
// CHECK-EL-SF-64R2-64: "-internal-isystem"
// CHECK-EL-SF-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-SF-64R2-64: "-internal-isystem"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el/sof"
// CHECK-EL-SF-64R2-64: "-internal-isystem"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-SF-64R2-64: "-internal-externc-isystem"
// CHECK-EL-SF-64R2-64: "[[TC]]/include"
// CHECK-EL-SF-64R2-64: "-internal-externc-isystem"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-SF-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-SF-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el/sof"
// CHECK-EL-SF-64R2-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/sof/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/sof/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-SF-64R2-64: "[[TC]]/mips64r2/64/el/sof{{/|\\\\}}crtbegin.o"
// CHECK-EL-SF-64R2-64: "-L[[TC]]/mips64r2/64/el/sof"
// CHECK-EL-SF-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el/sof"
// CHECK-EL-SF-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/sof/usr/lib"
// CHECK-EL-SF-64R2-64: "[[TC]]/mips64r2/64/el/sof{{/|\\\\}}crtend.o"
// CHECK-EL-SF-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/sof/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI 64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN-64R2-64 %s
// CHECK-EL-NAN-64R2-64: "-internal-isystem"
// CHECK-EL-NAN-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN-64R2-64: "-internal-isystem"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el/nan2008"
// CHECK-EL-NAN-64R2-64: "-internal-isystem"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN-64R2-64: "-internal-externc-isystem"
// CHECK-EL-NAN-64R2-64: "[[TC]]/include"
// CHECK-EL-NAN-64R2-64: "-internal-externc-isystem"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008"
// CHECK-EL-NAN-64R2-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN-64R2-64: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN-64R2-64: "-L[[TC]]/mips64r2/64/el/nan2008"
// CHECK-EL-NAN-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el/nan2008"
// CHECK-EL-NAN-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib"
// CHECK-EL-NAN-64R2-64: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, mips64r2, ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mips64r2 -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-64R2-64 %s
// CHECK-EL-NAN64-64R2-64: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-64R2-64: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-64R2-64: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/include"
// CHECK-EL-NAN64-64R2-64: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-64R2-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-64R2-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-64R2-64: "-L[[TC]]/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-64R2-64: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// = Little-endian, default (mips64r2), ABI 64, fp64, nan2008
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-mti-linux-gnu -mabi=64 -mfp64 -mnan=2008 \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-NAN64-64R2-64-DEF %s
// CHECK-EL-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-EL-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64-DEF: "-internal-isystem"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-EL-NAN64-64R2-64-DEF: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/include"
// CHECK-EL-NAN64-64R2-64-DEF: "-internal-externc-isystem"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-EL-NAN64-64R2-64-DEF: "{{.*}}ld{{(.exe)?}}"
// CHECK-EL-NAN64-64R2-64-DEF: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64-DEF: "-dynamic-linker" "/lib64/ld-linux-mipsn8.so.1"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crt1.o"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crti.o"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtbegin.o"
// CHECK-EL-NAN64-64R2-64-DEF: "-L[[TC]]/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64-DEF: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64/el/nan2008"
// CHECK-EL-NAN64-64R2-64-DEF: "-L[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/mips64r2/64/el/nan2008{{/|\\\\}}crtend.o"
// CHECK-EL-NAN64-64R2-64-DEF: "[[TC]]/../../../../sysroot/mips64r2/64/el/nan2008/usr/lib{{/|\\\\}}crtn.o"
//
// Check that mips32r3 and mips32r5 are equal to mips32r2
//
// = Big-endian, mips32r3, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r3 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32R3 %s
// CHECK-BE-HF-32R3: "-internal-isystem"
// CHECK-BE-HF-32R3: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-32R3: "-internal-isystem"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu"
// CHECK-BE-HF-32R3: "-internal-isystem"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-32R3: "-internal-externc-isystem"
// CHECK-BE-HF-32R3: "[[TC]]/include"
// CHECK-BE-HF-32R3: "-internal-externc-isystem"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-32R3: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32R3: "--sysroot=[[TC]]/../../../../sysroot"
// CHECK-BE-HF-32R3: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32R3: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-32R3: "-L[[TC]]"
// CHECK-BE-HF-32R3: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib"
// CHECK-BE-HF-32R3: "-L[[TC]]/../../../../sysroot/usr/lib/../lib"
// CHECK-BE-HF-32R3: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-HF-32R3: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips32r5, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux-gnu -mips32r5 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32R5 %s
// CHECK-BE-HF-32R5: "-internal-isystem"
// CHECK-BE-HF-32R5: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-32R5: "-internal-isystem"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu"
// CHECK-BE-HF-32R5: "-internal-isystem"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-32R5: "-internal-externc-isystem"
// CHECK-BE-HF-32R5: "[[TC]]/include"
// CHECK-BE-HF-32R5: "-internal-externc-isystem"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-32R5: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-32R5: "--sysroot=[[TC]]/../../../../sysroot"
// CHECK-BE-HF-32R5: "-dynamic-linker" "/lib/ld.so.1"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32R5: "[[TC]]{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-32R5: "-L[[TC]]"
// CHECK-BE-HF-32R5: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/../lib"
// CHECK-BE-HF-32R5: "-L[[TC]]/../../../../sysroot/usr/lib/../lib"
// CHECK-BE-HF-32R5: "[[TC]]{{/|\\\\}}crtend.o"
// CHECK-BE-HF-32R5: "[[TC]]/../../../../sysroot/usr/lib/../lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r3, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r3 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64R3-64 %s
// CHECK-BE-HF-64R3-64: "-internal-isystem"
// CHECK-BE-HF-64R3-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64R3-64: "-internal-isystem"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64"
// CHECK-BE-HF-64R3-64: "-internal-isystem"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64R3-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R3-64: "[[TC]]/include"
// CHECK-BE-HF-64R3-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64R3-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64R3-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64"
// CHECK-BE-HF-64R3-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64R3-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64R3-64: "-L[[TC]]/mips64r2/64"
// CHECK-BE-HF-64R3-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64"
// CHECK-BE-HF-64R3-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/usr/lib"
// CHECK-BE-HF-64R3-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64R3-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crtn.o"
//
// = Big-endian, mips64r5, ABI 64, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-mti-linux-gnu -mips64r5 -mabi=64 -mhard-float \
// RUN:     -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/mips_fsf_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64R5-64 %s
// CHECK-BE-HF-64R5-64: "-internal-isystem"
// CHECK-BE-HF-64R5-64: "[[TC:[^"]+/lib/gcc/mips-mti-linux-gnu/4.9.0]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0"
// CHECK-BE-HF-64R5-64: "-internal-isystem"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/mips-mti-linux-gnu/mips64r2/64"
// CHECK-BE-HF-64R5-64: "-internal-isystem"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../mips-mti-linux-gnu/include/c++/4.9.0/backward"
// CHECK-BE-HF-64R5-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R5-64: "[[TC]]/include"
// CHECK-BE-HF-64R5-64: "-internal-externc-isystem"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../sysroot/usr/include"
// CHECK-BE-HF-64R5-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-BE-HF-64R5-64: "--sysroot=[[TC]]/../../../../sysroot/mips64r2/64"
// CHECK-BE-HF-64R5-64: "-dynamic-linker" "/lib64/ld.so.1"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-64R5-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtbegin.o"
// CHECK-BE-HF-64R5-64: "-L[[TC]]/mips64r2/64"
// CHECK-BE-HF-64R5-64: "-L[[TC]]/../../../../mips-mti-linux-gnu/lib/mips64r2/64"
// CHECK-BE-HF-64R5-64: "-L[[TC]]/../../../../sysroot/mips64r2/64/usr/lib"
// CHECK-BE-HF-64R5-64: "[[TC]]/mips64r2/64{{/|\\\\}}crtend.o"
// CHECK-BE-HF-64R5-64: "[[TC]]/../../../../sysroot/mips64r2/64/usr/lib{{/|\\\\}}crtn.o"
