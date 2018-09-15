// General tests that ld invocations on Linux targets sane. Note that we use
// sysroot to make these tests independent of the host system.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32-NOT: warning:
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-LD-32: "-L[[SYSROOT]]/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64-NOT: warning:
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64: "--eh-frame-hdr"
// CHECK-LD-64: "-m" "elf_x86_64"
// CHECK-LD-64: "-dynamic-linker"
// CHECK-LD-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-64: "-L[[SYSROOT]]/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-64: "-lc"
// CHECK-LD-64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32 %s
// CHECK-LD-X32-NOT: warning:
// CHECK-LD-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-X32: "--eh-frame-hdr"
// CHECK-LD-X32: "-m" "elf32_x86_64"
// CHECK-LD-X32: "-dynamic-linker"
// CHECK-LD-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-X32: "-lc"
// CHECK-LD-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT %s
// CHECK-LD-RT-NOT: warning:
// CHECK-LD-RT: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT: "--eh-frame-hdr"
// CHECK-LD-RT: "-m" "elf_x86_64"
// CHECK-LD-RT: "-dynamic-linker"
// CHECK-LD-RT: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-RT: "-L[[SYSROOT]]/lib"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-RT: libclang_rt.builtins-x86_64.a"
// CHECK-LD-RT: "-lc"
// CHECK-LD-RT: libclang_rt.builtins-x86_64.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT-I686 %s
// CHECK-LD-RT-I686-NOT: warning:
// CHECK-LD-RT-I686: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT-I686: "--eh-frame-hdr"
// CHECK-LD-RT-I686: "-m" "elf_i386"
// CHECK-LD-RT-I686: "-dynamic-linker"
// CHECK-LD-RT-I686: "{{.*}}/usr/lib/gcc/i686-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib/gcc/i686-unknown-linux/4.6.0"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib/gcc/i686-unknown-linux/4.6.0/../../../../i686-unknown-linux/lib"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib/gcc/i686-unknown-linux/4.6.0/../../.."
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/lib"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-RT-I686: libclang_rt.builtins-i386.a"
// CHECK-LD-RT-I686: "-lc"
// CHECK-LD-RT-I686: libclang_rt.builtins-i386.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT-ANDROID %s
// CHECK-LD-RT-ANDROID-NOT: warning:
// CHECK-LD-RT-ANDROID: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT-ANDROID: "--eh-frame-hdr"
// CHECK-LD-RT-ANDROID: "-m" "armelf_linux_eabi"
// CHECK-LD-RT-ANDROID: "-dynamic-linker"
// CHECK-LD-RT-ANDROID: libclang_rt.builtins-arm-android.a"
// CHECK-LD-RT-ANDROID: "-lc"
// CHECK-LD-RT-ANDROID: libclang_rt.builtins-arm-android.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=libgcc \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GCC %s
// CHECK-LD-GCC-NOT: warning:
// CHECK-LD-GCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GCC: "--eh-frame-hdr"
// CHECK-LD-GCC: "-m" "elf_x86_64"
// CHECK-LD-GCC: "-dynamic-linker"
// CHECK-LD-GCC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-GCC: "-L[[SYSROOT]]/lib"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-GCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GCC: "-lc"
// CHECK-LD-GCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC-LIBGCC %s
// CHECK-LD-64-STATIC-LIBGCC-NOT: warning:
// CHECK-LD-64-STATIC-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64-STATIC-LIBGCC: "--eh-frame-hdr"
// CHECK-LD-64-STATIC-LIBGCC: "-m" "elf_x86_64"
// CHECK-LD-64-STATIC-LIBGCC: "-dynamic-linker"
// CHECK-LD-64-STATIC-LIBGCC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
// CHECK-LD-64-STATIC-LIBGCC: "-lc"
// CHECK-LD-64-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     -static \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC %s
// CHECK-LD-64-STATIC-NOT: warning:
// CHECK-LD-64-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64-STATIC: "--eh-frame-hdr"
// CHECK-LD-64-STATIC: "-m" "elf_x86_64"
// CHECK-LD-64-STATIC-NOT: "-dynamic-linker"
// CHECK-LD-64-STATIC: "-static"
// CHECK-LD-64-STATIC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbeginT.o"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/lib"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64-STATIC: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"
//
// Check that flags can be combined. The -static dominates.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     -static-libgcc -static \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC %s
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-32 %s
// CHECK-32-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-64 %s
// CHECK-32-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-64: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0/64{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-64 %s
// CHECK-64-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-32 %s
// CHECK-64-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32 %s
// CHECK-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../libx32"
// CHECK-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-X32: "-L[[SYSROOT]]/lib"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -mx32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-X32 %s
// CHECK-64-TO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-64-TO-X32: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -mx32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-X32 %s
// CHECK-32-TO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/x32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-32-TO-X32: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32-TO-64 %s
// CHECK-X32-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32-TO-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-X32-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32-TO-32 %s
// CHECK-X32-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32-TO-32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32{{/|\\\\}}crtbegin.o"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-X32-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -m32 \
// RUN:     --gcc-toolchain=%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-32-SYSROOT %s
// CHECK-64-TO-32-SYSROOT: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-32-SYSROOT: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-32-SYSROOT: "-L{{[^"]*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-64-TO-32-SYSROOT: "-L{{[^"]*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/fake_install_tree/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INSTALL-DIR-32 %s
// CHECK-INSTALL-DIR-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-INSTALL-DIR-32: "{{.*}}/Inputs/fake_install_tree/bin/../lib/gcc/i386-unknown-linux/4.7.0{{/|\\\\}}crtbegin.o"
// CHECK-INSTALL-DIR-32: "-L{{.*}}/Inputs/fake_install_tree/bin/../lib/gcc/i386-unknown-linux/4.7.0"
//
// Check that with 64-bit builds, we don't actually use the install directory
// as its version of GCC is lower than our sysrooted version.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -m64 \
// RUN:     -ccc-install-dir %S/Inputs/fake_install_tree/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INSTALL-DIR-64 %s
// CHECK-INSTALL-DIR-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-INSTALL-DIR-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtbegin.o"
// CHECK-INSTALL-DIR-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
//
// Check that we support unusual patch version formats, including missing that
// component.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing1/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION1 %s
// CHECK-GCC-VERSION1: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION1: "{{.*}}/Inputs/gcc_version_parsing1/bin/../lib/gcc/i386-unknown-linux/4.7{{/|\\\\}}crtbegin.o"
// CHECK-GCC-VERSION1: "-L{{.*}}/Inputs/gcc_version_parsing1/bin/../lib/gcc/i386-unknown-linux/4.7"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing2/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION2 %s
// CHECK-GCC-VERSION2: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION2: "{{.*}}/Inputs/gcc_version_parsing2/bin/../lib/gcc/i386-unknown-linux/4.7.x{{/|\\\\}}crtbegin.o"
// CHECK-GCC-VERSION2: "-L{{.*}}/Inputs/gcc_version_parsing2/bin/../lib/gcc/i386-unknown-linux/4.7.x"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing3/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION3 %s
// CHECK-GCC-VERSION3: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION3: "{{.*}}/Inputs/gcc_version_parsing3/bin/../lib/gcc/i386-unknown-linux/4.7.99-rc5{{/|\\\\}}crtbegin.o"
// CHECK-GCC-VERSION3: "-L{{.*}}/Inputs/gcc_version_parsing3/bin/../lib/gcc/i386-unknown-linux/4.7.99-rc5"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing4/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION4 %s
// CHECK-GCC-VERSION4: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION4: "{{.*}}/Inputs/gcc_version_parsing4/bin/../lib/gcc/i386-unknown-linux/4.7.99{{/|\\\\}}crtbegin.o"
// CHECK-GCC-VERSION4: "-L{{.*}}/Inputs/gcc_version_parsing4/bin/../lib/gcc/i386-unknown-linux/4.7.99"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing5/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION5 %s
// CHECK-GCC-VERSION5: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION5: "{{.*}}/Inputs/gcc_version_parsing5/bin/../lib/gcc/i386-unknown-linux/5{{/|\\\\}}crtbegin.o"
// CHECK-GCC-VERSION5: "-L{{.*}}/Inputs/gcc_version_parsing5/bin/../lib/gcc/i386-unknown-linux/5"
//
// Test a simulated installation of libc++ on Linux, both through sysroot and
// the installation path of Clang.
// RUN: %clangxx -no-canonical-prefixes -x c++ %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-SYSROOT %s
// CHECK-BASIC-LIBCXX-SYSROOT: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-SYSROOT: "--sysroot=[[SYSROOT]]"
// RUN: %clang -no-canonical-prefixes -x c++ %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-INSTALL %s
// CHECK-BASIC-LIBCXX-INSTALL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-INSTALL: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v1"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-INSTALL: "--sysroot=[[SYSROOT]]"
// CHECK-BASIC-LIBCXX-INSTALL: "-L[[SYSROOT]]/usr/bin/../lib"
//
// Test that we can use -stdlib=libc++ in a build system even when it
// occasionally links C code instead of C++ code.
// RUN: %clang -no-canonical-prefixes -x c %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-C-LINK %s
// CHECK-BASIC-LIBCXX-C-LINK-NOT: warning:
// CHECK-BASIC-LIBCXX-C-LINK: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-C-LINK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-C-LINK-NOT: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v1"
// CHECK-BASIC-LIBCXX-C-LINK: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-C-LINK: "--sysroot=[[SYSROOT]]"
// CHECK-BASIC-LIBCXX-C-LINK: "-L[[SYSROOT]]/usr/bin/../lib"
//
// Test a very broken version of multiarch that shipped in Ubuntu 11.04.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_11.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-11-04 %s
// CHECK-UBUNTU-11-04: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-11-04: "{{.*}}/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5"
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../i386-linux-gnu"
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/usr/lib/i386-linux-gnu"
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../.."
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/lib"
// CHECK-UBUNTU-11-04: "-L[[SYSROOT]]/usr/lib"
//
// Check multi arch support on Ubuntu 12.04 LTS.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-unknown-linux-gnueabihf \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_12.04_LTS_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-12-04-ARM-HF %s
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/../../../arm-linux-gnueabihf{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/../../../arm-linux-gnueabihf{{/|\\\\}}crti.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabihf/4.6.3"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/../../../arm-linux-gnueabihf"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/lib/arm-linux-gnueabihf"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/arm-linux-gnueabihf"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/../../.."
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3/../../../arm-linux-gnueabihf{{/|\\\\}}crtn.o"
//
// Check Ubuntu 13.10 on x86-64 targeting arm-linux-gnueabihf.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabihf \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/x86-64_ubuntu_13.10 \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64-UBUNTU-13-10-ARM-HF %s
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-dynamic-linker" "{{(/usr/arm--linux-gnueabihf)?}}/lib/ld-linux-armhf.so.3"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crti.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8{{/|\\\\}}crtbegin.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8{{/|\\\\}}crtend.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crtn.o"
//
// Check Ubuntu 13.10 on x86-64 targeting arm-linux-gnueabi.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/x86-64_ubuntu_13.10 \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64-UBUNTU-13-10-ARM %s
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-dynamic-linker" "{{(/usr/arm--linux-gnueabi)?}}/lib/ld-linux.so.3"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crti.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7{{/|\\\\}}crtbegin.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7{{/|\\\\}}crtend.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crtn.o"
//
// Check Ubuntu 14.04 on powerpc64le.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-unknown-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-PPC64LE %s
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../powerpc64le-linux-gnu{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../powerpc64le-linux-gnu{{/|\\\\}}crti.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/lib/powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../.."
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../powerpc64le-linux-gnu{{/|\\\\}}crtn.o"
//
// Check Ubuntu 14.04 on x32.
// "/usr/lib/gcc/x86_64-linux-gnu/4.8/x32/crtend.o" "/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32/crtn.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-X32 %s
// CHECK-UBUNTU-14-04-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crti.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/x32{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/x32"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/x86_64-linux-gnu/../../libx32"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../.."
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/x32{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crtn.o"
//
// Check fedora 18 on arm.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-unknown-linux-gnueabihf \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_18_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-18-ARM-HF %s
// CHECK-FEDORA-18-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crti.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2{{/|\\\\}}crtbegin.o"
// CHECK-FEDORA-18-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2"
// CHECK-FEDORA-18-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2{{/|\\\\}}crtend.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crtn.o"
//
// Check Fedora 21 on AArch64.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_21_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-21-AARCH64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_21_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-21-AARCH64 %s
// CHECK-FEDORA-21-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0{{/|\\\\}}crtbegin.o"
// CHECK-FEDORA-21-AARCH64: "-L[[SYSROOT]]/usr/lib/gcc/aarch64-redhat-linux/4.9.0"
// CHECK-FEDORA-21-AARCH64: "-L[[SYSROOT]]/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0{{/|\\\\}}crtend.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-unknown-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_12.04_LTS_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-12-04-ARM %s
// CHECK-UBUNTU-12-04-ARM: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1/../../../arm-linux-gnueabi{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1/../../../arm-linux-gnueabi{{/|\\\\}}crti.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabi/4.6.1"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabi/4.6.1/../../../arm-linux-gnueabi"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/lib/arm-linux-gnueabi"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/arm-linux-gnueabi"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabi/4.6.1/../../.."
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1/../../../arm-linux-gnueabi{{/|\\\\}}crtn.o"
//
// Test the setup that shipped in SUSE 10.3 on ppc64.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-suse-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/suse_10.3_ppc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SUSE-10-3-PPC64 %s
// CHECK-SUSE-10-3-PPC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SUSE-10-3-PPC64: "{{.*}}/usr/lib/gcc/powerpc64-suse-linux/4.1.2/64{{/|\\\\}}crtbegin.o"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-suse-linux/4.1.2/64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-suse-linux/4.1.2/../../../../lib64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/../lib64"
//
// Check openSuse Leap 42.2 on AArch64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_42.2_aarch64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-42-2-AARCH64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_42.2_aarch64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-42-2-AARCH64 %s
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-42-2-AARCH64: "-L[[SYSROOT]]/usr/lib64/gcc/aarch64-suse-linux/4.8"
// CHECK-OPENSUSE-42-2-AARCH64: "-L[[SYSROOT]]/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on armv6hl
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv6hl-suse-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv6hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV6HL %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv6hl-suse-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv6hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV6HL %s
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "-L[[SYSROOT]]/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5"
// CHECK-OPENSUSE-TW-ARMV6HL: "-L[[SYSROOT]]/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on armv7hl
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7hl-suse-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv7hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV7HL %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7hl-suse-linux-gnueabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv7hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV7HL %s
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5"
// CHECK-OPENSUSE-TW-ARMV7HL: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crtn.o"
//
// Check dynamic-linker for different archs
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM: "-m" "armelf_linux_eabi"
// CHECK-ARM: "-dynamic-linker" "{{.*}}/lib/ld-linux.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-ABIHF %s
// CHECK-ARM-ABIHF: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM-ABIHF: "-m" "armelf_linux_eabi"
// CHECK-ARM-ABIHF: "-dynamic-linker" "{{.*}}/lib/ld-linux-armhf.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabihf \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-HF %s
// CHECK-ARM-HF: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM-HF: "-m" "armelf_linux_eabi"
// CHECK-ARM-HF: "-dynamic-linker" "{{.*}}/lib/ld-linux-armhf.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64 %s
// CHECK-PPC64: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64: "-m" "elf64ppc"
// CHECK-PPC64: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu -mabi=elfv1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64-ELFv1 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu -mabi=elfv1-qpx \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64-ELFv1 %s
// CHECK-PPC64-ELFv1: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64-ELFv1: "-m" "elf64ppc"
// CHECK-PPC64-ELFv1: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu -mabi=elfv2 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64-ELFv2 %s
// CHECK-PPC64-ELFv2: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64-ELFv2: "-m" "elf64ppc"
// CHECK-PPC64-ELFv2: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE %s
// CHECK-PPC64LE: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE: "-m" "elf64lppc"
// CHECK-PPC64LE: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu -mabi=elfv1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE-ELFv1 %s
// CHECK-PPC64LE-ELFv1: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE-ELFv1: "-m" "elf64lppc"
// CHECK-PPC64LE-ELFv1: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu -mabi=elfv2 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE-ELFv2 %s
// CHECK-PPC64LE-ELFv2: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE-ELFv2: "-m" "elf64lppc"
// CHECK-PPC64LE-ELFv2: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// Check that we do not pass --hash-style=gnu or --hash-style=both to
// hexagon linux linker
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=hexagon-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-HEXAGON %s
// CHECK-HEXAGON: "{{.*}}hexagon-link{{(.exe)?}}"
// CHECK-HEXAGON-NOT: "--hash-style={{gnu|both}}"
//
// Check that we do not pass --hash-style=gnu and --hash-style=both to linker
// and provide correct path to the dynamic linker and emulation mode when build
// for MIPS platforms.
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS: "-m" "elf32btsmip"
// CHECK-MIPS: "-dynamic-linker" "{{.*}}/lib/ld.so.1"
// CHECK-MIPS-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL %s
// CHECK-MIPSEL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPSEL: "-m" "elf32ltsmip"
// CHECK-MIPSEL: "-dynamic-linker" "{{.*}}/lib/ld.so.1"
// CHECK-MIPSEL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-gnu -mnan=2008 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL-NAN2008 %s
// CHECK-MIPSEL-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPSEL-NAN2008: "-m" "elf32ltsmip"
// CHECK-MIPSEL-NAN2008: "-dynamic-linker" "{{.*}}/lib/ld-linux-mipsn8.so.1"
// CHECK-MIPSEL-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-gnu -mcpu=mips32r6 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS32R6EL %s
// CHECK-MIPS32R6EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS32R6EL: "-m" "elf32ltsmip"
// CHECK-MIPS32R6EL: "-dynamic-linker" "{{.*}}/lib/ld-linux-mipsn8.so.1"
// CHECK-MIPS32R6EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64 %s
// CHECK-MIPS64: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64: "-m" "elf64btsmip"
// CHECK-MIPS64: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL %s
// CHECK-MIPS64EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL: "-m" "elf64ltsmip"
// CHECK-MIPS64EL: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mnan=2008 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-NAN2008 %s
// CHECK-MIPS64EL-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-NAN2008: "-m" "elf64ltsmip"
// CHECK-MIPS64EL-NAN2008: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64EL-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mcpu=mips64r6 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64R6EL %s
// CHECK-MIPS64R6EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64R6EL: "-m" "elf64ltsmip"
// CHECK-MIPS64R6EL: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64R6EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu -mabi=n32 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-N32 %s
// CHECK-MIPS64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64-N32: "-m" "elf32btsmipn32"
// CHECK-MIPS64-N32: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld.so.1"
// CHECK-MIPS64-N32-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -mabi=n32 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-N32 %s
// CHECK-MIPS64EL-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-N32: "-m" "elf32ltsmipn32"
// CHECK-MIPS64EL-N32: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld.so.1"
// CHECK-MIPS64EL-N32-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mabi=n32 \
// RUN:   -mnan=2008 | FileCheck --check-prefix=CHECK-MIPS64EL-N32-NAN2008 %s
// CHECK-MIPS64EL-N32-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-N32-NAN2008: "-m" "elf32ltsmipn32"
// CHECK-MIPS64EL-N32-NAN2008: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64EL-N32-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-redhat-linux \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-REDHAT %s
// CHECK-MIPS64EL-REDHAT: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-REDHAT: "-m" "elf64ltsmip"
// CHECK-MIPS64EL-REDHAT: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64EL-REDHAT-NOT: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-musl-mipsel.so.1"
// CHECK-MIPS64EL-REDHAT-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV8 %s
// CHECK-SPARCV8: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV8: "-m" "elf32_sparc"
// CHECK-SPARCV8: "-dynamic-linker" "{{(/usr/sparc-unknown-linux-gnu)?}}/lib/ld-linux.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparcel-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV8EL %s
// CHECK-SPARCV8EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV8EL: "-m" "elf32_sparc"
// CHECK-SPARCV8EL: "-dynamic-linker" "{{(/usr/sparcel-unknown-linux-gnu)?}}/lib/ld-linux.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparcv9-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV9 %s
// CHECK-SPARCV9: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV9: "-m" "elf64_sparc"
// CHECK-SPARCV9: "-dynamic-linker" "{{(/usr/sparcv9-unknown-linux-gnu)?}}/lib{{(64)?}}/ld-linux.so.2"
//
// Thoroughly exercise the Debian multiarch environment.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86 %s
// CHECK-DEBIAN-X86: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86: "{{.*}}/usr/lib/gcc/i686-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5"
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../i386-linux-gnu"
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/usr/lib/i386-linux-gnu"
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-X86: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86-64 %s
// CHECK-DEBIAN-X86-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86-64: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5"
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/usr/lib/x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-X86-64: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC %s
// CHECK-DEBIAN-PPC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC: "{{.*}}/usr/lib/gcc/powerpc-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5"
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/usr/lib/powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-PPC: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC64LE %s
// CHECK-DEBIAN-PPC64LE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.5"
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.5/../../../powerpc64le-linux-gnu"
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/usr/lib/powerpc64le-linux-gnu"
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-PPC64LE: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC64 %s
// CHECK-DEBIAN-PPC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC64: "{{.*}}/usr/lib/gcc/powerpc64-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5"
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/usr/lib/powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-PPC64: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS %s
// CHECK-DEBIAN-MIPS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS: "{{.*}}/usr/lib/gcc/mips-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/../../../mips-linux-gnu"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/mips-linux-gnu"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPSEL %s
// CHECK-DEBIAN-MIPSEL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.5{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/../../../mipsel-linux-gnu"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/mipsel-linux-gnu"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPSEL: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS64 %s
// CHECK-DEBIAN-MIPS64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS64: "{{.*}}/usr/lib/gcc/mips-linux-gnu/4.5/64{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS64: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/64"
// CHECK-DEBIAN-MIPS64: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5"
// CHECK-DEBIAN-MIPS64: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPS64: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS64: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS64EL %s
// CHECK-DEBIAN-MIPS64EL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.5/64{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/64"
// CHECK-DEBIAN-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5"
// CHECK-DEBIAN-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPS64EL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS64EL: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu -mabi=n32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS64-N32 %s
// CHECK-DEBIAN-MIPS64-N32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS64-N32: "{{.*}}/usr/lib/gcc/mips-linux-gnu/4.5/n32{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS64-N32: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/n32"
// CHECK-DEBIAN-MIPS64-N32: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5"
// CHECK-DEBIAN-MIPS64-N32: "-L[[SYSROOT]]/usr/lib/gcc/mips-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPS64-N32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS64-N32: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -mabi=n32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-MIPS64EL-N32 %s
// CHECK-DEBIAN-MIPS64EL-N32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.5/n32{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/n32"
// CHECK-DEBIAN-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5"
// CHECK-DEBIAN-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.5/../../.."
// CHECK-DEBIAN-MIPS64EL-N32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib"
//
// Check linker paths on Debian 8 / Sparc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc_multilib_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC32 %s
// CHECK-DEBIAN-SPARC32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC32: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../sparc-linux-gnu{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-SPARC32: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../sparc-linux-gnu{{/|\\\\}}crti.o"
// CHECK-DEBIAN-SPARC32: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../sparc-linux-gnu"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../lib"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/lib/sparc-linux-gnu"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/usr/lib/sparc-linux-gnu"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-SPARC32: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-SPARC32: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-SPARC32: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../sparc-linux-gnu{{/|\\\\}}crtn.o"
//
// Check linker paths on Debian 8 / Sparc, with the oldstyle multilib packages
// RUN: %clang -no-canonical-prefixes -m64 %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc_multilib_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC32-LIB64 %s
// CHECK-DEBIAN-SPARC32-LIB64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC32-LIB64: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-SPARC32-LIB64: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-SPARC32-LIB64: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/64{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/64"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../lib64"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-SPARC32-LIB64: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-SPARC32-LIB64: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/64{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-SPARC32-LIB64: "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../lib64{{/|\\\\}}crtn.o"
//
// Check linker paths on Debian 8 / Sparc64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc64-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC64 %s
// CHECK-DEBIAN-SPARC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC64: "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../sparc64-linux-gnu{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-SPARC64: "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../sparc64-linux-gnu{{/|\\\\}}crti.o"
// CHECK-DEBIAN-SPARC64: "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../sparc64-linux-gnu"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/lib/sparc64-linux-gnu"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/usr/lib/sparc64-linux-gnu"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../.."
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-SPARC64: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-SPARC64: "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-SPARC64: "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../sparc64-linux-gnu{{/|\\\\}}crtn.o"
//
// Test linker invocation on Android.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// CHECK-ANDROID: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID: "--enable-new-dtags"
// CHECK-ANDROID: "{{.*}}{{/|\\\\}}crtbegin_dynamic.o"
// CHECK-ANDROID: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-NOT: "gcc_s"
// CHECK-ANDROID: "-lgcc"
// CHECK-ANDROID: "-ldl"
// CHECK-ANDROID-NOT: "gcc_s"
// CHECK-ANDROID: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// CHECK-ANDROID-SO: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-SO-NOT: "-Bsymbolic"
// CHECK-ANDROID-SO: "{{.*}}{{/|\\\\}}crtbegin_so.o"
// CHECK-ANDROID-SO: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-SO-NOT: "gcc_s"
// CHECK-ANDROID-SO: "-lgcc"
// CHECK-ANDROID-SO: "-ldl"
// CHECK-ANDROID-SO-NOT: "gcc_s"
// CHECK-ANDROID-SO: "{{.*}}{{/|\\\\}}crtend_so.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// CHECK-ANDROID-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-STATIC: "{{.*}}{{/|\\\\}}crtbegin_static.o"
// CHECK-ANDROID-STATIC: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-STATIC-NOT: "gcc_s"
// CHECK-ANDROID-STATIC: "-lgcc"
// CHECK-ANDROID-STATIC-NOT: "-ldl"
// CHECK-ANDROID-STATIC-NOT: "gcc_s"
// CHECK-ANDROID-STATIC: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// CHECK-ANDROID-PIE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-PIE: "{{.*}}{{/|\\\\}}crtbegin_dynamic.o"
// CHECK-ANDROID-PIE: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-PIE-NOT: "gcc_s"
// CHECK-ANDROID-PIE: "-lgcc"
// CHECK-ANDROID-PIE-NOT: "gcc_s"
// CHECK-ANDROID-PIE: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// CHECK-ANDROID-32: "-dynamic-linker" "/system/bin/linker"
// CHECK-ANDROID-64: "-dynamic-linker" "/system/bin/linker64"
//
// Test that -pthread does not add -lpthread on Android.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -pthread \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// CHECK-ANDROID-PTHREAD-NOT: -lpthread
//
// RUN: %clang -no-canonical-prefixes %t.o -### -o %t 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD-LINK %s
// CHECK-ANDROID-PTHREAD-LINK-NOT: argument unused during compilation: '-pthread'
//
// Check linker invocation on Debian 6 MIPS 32/64-bit.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPSEL %s
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/lib/../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../.."
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL %s
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/64{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../.."
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -mabi=n32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-N32 %s
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/n32{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/n32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../.."
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnuabi64 -mabi=n64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64-GNUABI %s
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../mips64-linux-gnuabi64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../mips64-linux-gnuabi64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../mips64-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/lib/mips64-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/mips64-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../.."
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../mips64-linux-gnuabi64{{/|\\\\}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnuabi64 -mabi=n64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-GNUABI %s
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../mips64el-linux-gnuabi64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../mips64el-linux-gnuabi64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../mips64el-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/lib/mips64el-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/mips64el-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../.."
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../mips64el-linux-gnuabi64{{/|\\\\}}crtn.o"
//
// Test linker invocation for Freescale SDK (OpenEmbedded).
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-fsl-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/freescale_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FSL-PPC %s
// CHECK-FSL-PPC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FSL-PPC: "-m" "elf32ppclinux"
// CHECK-FSL-PPC: "{{.*}}{{/|\\\\}}crt1.o"
// CHECK-FSL-PPC: "{{.*}}{{/|\\\\}}crtbegin.o"
// CHECK-FSL-PPC: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-fsl-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/freescale_ppc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FSL-PPC64 %s
// CHECK-FSL-PPC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FSL-PPC64: "-m" "elf64ppc"
// CHECK-FSL-PPC64: "{{.*}}{{/|\\\\}}crt1.o"
// CHECK-FSL-PPC64: "{{.*}}{{/|\\\\}}crtbegin.o"
// CHECK-FSL-PPC64: "-L[[SYSROOT]]/usr/lib64/powerpc64-fsl-linux/4.6.2/../.."
//
// Check that crtfastmath.o is linked with -ffast-math and with -Ofast.
// RUN: %clang --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -funsafe-math-optimizations\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -O3\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -O3 -Ofast\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -ffast-math -fno-fast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -fno-fast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -fno-unsafe-math-optimizations \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fno-fast-math -Ofast  \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fno-unsafe-math-optimizations -Ofast \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// We don't have crtfastmath.o in the i386 tree, use it to check that file
// detection works.
// RUN: %clang --target=i386-unknown-linux -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// CHECK-CRTFASTMATH: usr/lib/gcc/x86_64-unknown-linux/4.6.0{{/|\\\\}}crtfastmath.o
// CHECK-NOCRTFASTMATH-NOT: crtfastmath.o

// Check that we link in gcrt1.o when compiling with -pg
// RUN: %clang -pg --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>& 1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG %s
// CHECK-PG: gcrt1.o

// GCC forwards -u to the linker.
// RUN: %clang -u asdf --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>& 1 \
// RUN:   | FileCheck --check-prefix=CHECK-u %s
// CHECK-u: "-u" "asdf"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMEB %s
// CHECK-ARMEB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMEB-NOT: "--be8"
// CHECK-ARMEB: "-m" "armelfb_linux_eabi"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armebv7-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EB %s
// CHECK-ARMV7EB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMV7EB: "--be8"
// CHECK-ARMV7EB: "-m" "armelfb_linux_eabi"

// Check dynamic-linker for musl-libc
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-X86 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-X86_64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPSEL %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS64EL %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-PPC %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-PPC64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARM %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbv7-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbeb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEB %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbeb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbv7eb-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARM %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEB %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7eb-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-AARCH64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64_be-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-AARCH64_BE %s
// CHECK-MUSL-X86:        "-dynamic-linker" "/lib/ld-musl-i386.so.1"
// CHECK-MUSL-X86_64:     "-dynamic-linker" "/lib/ld-musl-x86_64.so.1"
// CHECK-MUSL-MIPS:       "-dynamic-linker" "/lib/ld-musl-mips.so.1"
// CHECK-MUSL-MIPSEL:     "-dynamic-linker" "/lib/ld-musl-mipsel.so.1"
// CHECK-MUSL-MIPS64:     "-dynamic-linker" "/lib/ld-musl-mips64.so.1"
// CHECK-MUSL-MIPS64EL:   "-dynamic-linker" "/lib/ld-musl-mips64el.so.1"
// CHECK-MUSL-PPC:        "-dynamic-linker" "/lib/ld-musl-powerpc.so.1"
// CHECK-MUSL-PPC64:      "-dynamic-linker" "/lib/ld-musl-powerpc64.so.1"
// CHECK-MUSL-ARM:        "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK-MUSL-ARMHF:      "-dynamic-linker" "/lib/ld-musl-armhf.so.1"
// CHECK-MUSL-ARMEB:      "-dynamic-linker" "/lib/ld-musl-armeb.so.1"
// CHECK-MUSL-ARMEBHF:    "-dynamic-linker" "/lib/ld-musl-armebhf.so.1"
// CHECK-MUSL-AARCH64:    "-dynamic-linker" "/lib/ld-musl-aarch64.so.1"
// CHECK-MUSL-AARCH64_BE: "-dynamic-linker" "/lib/ld-musl-aarch64_be.so.1"

// Check whether multilib gcc install works fine on Gentoo with gcc-config
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO %s
// CHECK-LD-GENTOO-NOT: warning:
// CHECK-LD-GENTOO: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO: "--eh-frame-hdr"
// CHECK-LD-GENTOO: "-m" "elf_x86_64"
// CHECK-LD-GENTOO: "-dynamic-linker"
// CHECK-LD-GENTOO: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3"
// CHECK-LD-GENTOO: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../.."
// CHECK-LD-GENTOO: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO: "-lc"
// CHECK-LD-GENTOO: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-unknown-linux-gnu -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO-32 %s
// CHECK-LD-GENTOO-32-NOT: warning:
// CHECK-LD-GENTOO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO-32: "--eh-frame-hdr"
// CHECK-LD-GENTOO-32: "-m" "elf_i386"
// CHECK-LD-GENTOO-32: "-dynamic-linker"
// CHECK-LD-GENTOO-32: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/32{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/32"
// CHECK-LD-GENTOO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../.."
// CHECK-LD-GENTOO-32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO-32: "-lc"
// CHECK-LD-GENTOO-32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO-X32 %s
// CHECK-LD-GENTOO-X32-NOT: warning:
// CHECK-LD-GENTOO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO-X32: "--eh-frame-hdr"
// CHECK-LD-GENTOO-X32: "-m" "elf32_x86_64"
// CHECK-LD-GENTOO-X32: "-dynamic-linker"
// CHECK-LD-GENTOO-X32: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/x32{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/x32"
// CHECK-LD-GENTOO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../.."
// CHECK-LD-GENTOO-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO-X32: "-lc"
// CHECK-LD-GENTOO-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="%S/Inputs/rhel_7_tree/opt/rh/devtoolset-7/root/usr" \
// RUN:     --sysroot=%S/Inputs/rhel_7_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RHEL7-DTS %s
// CHECK-LD-RHEL7-DTS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RHLE7-DTS: Selected GCC installation: [[GCC_INSTALL:[[SYSROOT]]/lib/gcc/x86_64-redhat-linux/7]]
// CHECK-LD-RHEL7-DTS-NOT: /usr/bin/ld
// CHECK-LD-RHLE7-DTS: [[GCC_INSTALL]/../../../bin/ld

// Check whether gcc7 install works fine on Amazon Linux AMI
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-amazon-linux -rtlib=libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ami_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-AMI %s
// CHECK-LD-AMI-NOT: warning:
// CHECK-LD-AMI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-AMI: "--eh-frame-hdr"
// CHECK-LD-AMI: "-m" "elf_x86_64"
// CHECK-LD-AMI: "-dynamic-linker"
// CHECK-LD-AMI: "{{.*}}/usr/lib/gcc/x86_64-amazon-linux/7{{/|\\\\}}crtbegin.o"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-amazon-linux/7"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-amazon-linux/7/../../../../lib64"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-amazon-linux/7/../../.."
// CHECK-LD-AMI: "-L[[SYSROOT]]/lib"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-AMI: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-AMI: "-lc"
// CHECK-LD-AMI: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// Check whether the OpenEmbedded ARM libs are added correctly.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-oe-linux-gnueabi -rtlib=libgcc \
// RUN:     --sysroot=%S/Inputs/openembedded_arm_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-ARM %s

// CHECK-OE-ARM: "-cc1" "-triple" "armv4t-oe-linux-gnueabi"
// CHECK-OE-ARM: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OE-ARM: "-m" "armelf_linux_eabi" "-dynamic-linker"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crt1.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crti.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0{{/|\\\\}}crtbegin.o"
// CHECK-OE-ARM: "-L[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0"
// CHECK-OE-ARM: "-L[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi"
// CHECK-OE-ARM: "-L[[SYSROOT]]/usr/lib"
// CHECK-OE-ARM: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0{{/|\\\\}}crtend.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crtn.o"

// Check whether the OpenEmbedded AArch64 libs are added correctly.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-oe-linux -rtlib=libgcc \
// RUN:     --sysroot=%S/Inputs/openembedded_aarch64_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-AARCH64 %s

// CHECK-OE-AARCH64: "-cc1" "-triple" "aarch64-oe-linux"
// CHECK-OE-AARCH64: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OE-AARCH64: "-m" "aarch64linux" "-dynamic-linker"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crt1.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crti.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0{{/|\\\\}}crtbegin.o"
// CHECK-OE-AARCH64: "-L[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0"
// CHECK-OE-AARCH64: "-L[[SYSROOT]]/usr/lib64"
// CHECK-OE-AARCH64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0{{/|\\\\}}crtend.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crtn.o"
