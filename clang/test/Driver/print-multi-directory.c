// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>/dev/null \
// RUN:     -target i386-none-linux \
// RUN:     -B%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:     -print-multi-directory \
// RUN:   | FileCheck --check-prefix=CHECK-X86-MULTILIBS %s

// CHECK-X86-MULTILIBS:      32
// CHECK-X86-MULTILIBS-NOT:  x32
// CHECK-X86-MULTILIBS-NOT:  .

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>/dev/null \
// RUN:     -target i386-none-linux -m64 \
// RUN:     -B%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:     -print-multi-directory \
// RUN:   | FileCheck --check-prefix=CHECK-X86_64-MULTILIBS %s

// CHECK-X86_64-MULTILIBS:      .
// CHECK-X86_64-MULTILIBS-NOT:  x32
// CHECK-X86_64-MULTILIBS-NOT:  32

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>/dev/null \
// RUN:     -target arm-linux-androideabi21 \
// RUN:     -mthumb \
// RUN:     -B%S/Inputs/basic_android_ndk_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_ndk_tree/sysroot \
// RUN:     -print-multi-directory \
// RUN:   | FileCheck  --check-prefix=CHECK-ARM-MULTILIBS %s

// CHECK-ARM-MULTILIBS:      thumb
// CHECK-ARM-MULTILIBS-NOT:  .
