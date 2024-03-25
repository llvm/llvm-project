// UNSUPPORTED: system-windows
// A basic clang -cc1 command-line, and simple environment check.

// RUN: %clang -### %s --target=csky 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: "-cc1" "-triple" "csky"

// In the below tests, --rtlib=platform is used so that the driver ignores
// the configure-time CLANG_DEFAULT_RTLIB option when choosing the runtime lib

// RUN: %clang -### %s -fuse-ld=ld -no-pie --target=csky-unknown-linux-gnu --rtlib=platform --unwindlib=platform \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_csky_linux_sdk  2>&1 | FileCheck -check-prefix=C-CSKY-LINUX-MULTI %s

// C-CSKY-LINUX-MULTI: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/bin{{/|\\\\}}ld"
// C-CSKY-LINUX-MULTI: "--sysroot={{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc"
// C-CSKY-LINUX-MULTI: "-m" "cskyelf_linux"
// C-CSKY-LINUX-MULTI: "-dynamic-linker" "/lib/ld.so.1"
// C-CSKY-LINUX-MULTI: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/usr/lib/../lib/crt1.o"
// C-CSKY-LINUX-MULTI: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/crti.o"
// C-CSKY-LINUX-MULTI: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/crtbegin.o"
// C-CSKY-LINUX-MULTI: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0"
// C-CSKY-LINUX-MULTI: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/lib/../lib"
// C-CSKY-LINUX-MULTI: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/usr/lib/../lib"
// C-CSKY-LINUX-MULTI: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/lib"
// C-CSKY-LINUX-MULTI: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/usr/lib"

// RUN: %clang -### %s -fuse-ld=ld -fno-pic -no-pie --target=csky-unknown-linux-gnu --rtlib=platform --unwindlib=platform -march=ck860v \
// RUN:   --gcc-toolchain=%S/Inputs/multilib_csky_linux_sdk 2>&1 | FileCheck -check-prefix=C-CSKY-LINUX-CK860V %s

// C-CSKY-LINUX-CK860V: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/bin{{/|\\\\}}ld"
// C-CSKY-LINUX-CK860V: "--sysroot={{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v"
// C-CSKY-LINUX-CK860V: "-m" "cskyelf_linux"
// C-CSKY-LINUX-CK860V: "-dynamic-linker" "/lib/ld.so.1"
// C-CSKY-LINUX-CK860V: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v/usr/lib/../lib/crt1.o"
// C-CSKY-LINUX-CK860V: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/ck860v/crti.o"
// C-CSKY-LINUX-CK860V: "{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/ck860v/crtbegin.o"
// C-CSKY-LINUX-CK860V: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/ck860v"
// C-CSKY-LINUX-CK860V: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v/lib/../lib"
// C-CSKY-LINUX-CK860V: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v/usr/lib/../lib"
// C-CSKY-LINUX-CK860V: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v/lib"
// C-CSKY-LINUX-CK860V: "-L{{.*}}/Inputs/multilib_csky_linux_sdk/lib/gcc/csky-linux-gnuabiv2/6.3.0/../../..{{/|\\\\}}..{{/|\\\\}}csky-linux-gnuabiv2/libc/ck860v/usr/lib"
