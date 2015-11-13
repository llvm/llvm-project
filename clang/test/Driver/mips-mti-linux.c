// Check frontend and linker invocations on GPL-free MIPS toolchain.
//
// FIXME: Using --sysroot with this toolchain/triple isn't supported. We use
//        it here to test that we are producing the correct paths/flags.
//        Ideally, we'd like to have an --llvm-toolchain option similar to
//        the --gcc-toolchain one.

// REQUIRES: shell

// = Big-endian, mips32r2, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux -mips32r2 -mhard-float \
// RUN:     --sysroot=%S/Inputs/mips_mti_linux/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32R2 %s
//
// CHECK-BE-HF-32R2: "{{[^"]*}}clang{{[^"]*}}" {{.*}} "-triple" "mips-mti-linux"
// CHECK-BE-HF-32R2-SAME: "-fuse-init-array" "-target-cpu" "mips32r2"
// CHECK-BE-HF-32R2-SAME: "-isysroot" "{{.*}}mips_mti_linux/sysroot"
// CHECK-BE-HF-32R2: "{{[^"]*}}lld{{[^"]*}}" "-flavor" "gnu" "-target" "mips-mti-linux"
// CHECK-BE-HF-32R2-SAME: "--sysroot=[[SYSROOT:[^"]+]]" {{.*}} "-dynamic-linker" "/lib/ld-musl-mips.so.1"
// CHECK-BE-HF-32R2-SAME: "[[SYSROOT]]/mips-r2-hard-musl/usr/lib{{/|\\\\}}crt1.o"
// CHECK-BE-HF-32R2-SAME: "[[SYSROOT]]/mips-r2-hard-musl/usr/lib{{/|\\\\}}crti.o"
// CHECK-BE-HF-32R2-SAME: "-L[[SYSROOT]]/mips-r2-hard-musl/usr/lib"
// CHECK-BE-HF-32R2-SAME: "{{[^"]+}}/mips-r2-hard-musl/lib/linux/libclang_rt.builtins-mips.a"
// CHECK-BE-HF-32R2-SAME: "-lc"
// CHECK-BE-HF-32R2-SAME: "[[SYSROOT]]/mips-r2-hard-musl/usr/lib{{/|\\\\}}crtn.o"

// = Little-endian, mips32r2, hard float
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-mti-linux -mips32r2 -EL -mhard-float \
// RUN:     --sysroot=%S/Inputs/mips_mti_linux/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-LE-HF-32R2 %s
//
// CHECK-LE-HF-32R2: "{{[^"]*}}clang{{[^"]*}}" {{.*}} "-triple" "mipsel-mti-linux"
// CHECK-LE-HF-32R2-SAME: "-fuse-init-array" "-target-cpu" "mips32r2"
// CHECK-LE-HF-32R2-SAME: "-isysroot" "{{.*}}mips_mti_linux/sysroot"
// CHECK-LE-HF-32R2: "{{[^"]*}}lld{{[^"]*}}" "-flavor" "gnu" "-target" "mipsel-mti-linux"
// CHECK-LE-HF-32R2-SAME: "--sysroot=[[SYSROOT:[^"]+]]" {{.*}} "-dynamic-linker" "/lib/ld-musl-mipsel.so.1"
// CHECK-LE-HF-32R2-SAME: "[[SYSROOT]]/mipsel-r2-hard-musl/usr/lib{{/|\\\\}}crt1.o"
// CHECK-LE-HF-32R2-SAME: "[[SYSROOT]]/mipsel-r2-hard-musl/usr/lib{{/|\\\\}}crti.o"
// CHECK-LE-HF-32R2-SAME: "-L[[SYSROOT]]/mipsel-r2-hard-musl/usr/lib"
// CHECK-LE-HF-32R2-SAME: "{{[^"]+}}/mipsel-r2-hard-musl/lib/linux/libclang_rt.builtins-mipsel.a"
// CHECK-LE-HF-32R2-SAME: "-lc"
// CHECK-LE-HF-32R2-SAME: "[[SYSROOT]]/mipsel-r2-hard-musl/usr/lib{{/|\\\\}}crtn.o"
