// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: warning:
// CHECK-DEFAULT:     "-C"
// CHECK-DEFAULT:     crt1.o
// CHECK-DEFAULT:     crti.o
// CHECK-DEFAULT:     values-Xa.o
// CHECK-DEFAULT:     values-xpg6.o
// CHECK-DEFAULT:     crtbegin.o
// CHECK-DEFAULT:     -lgcc
// CHECK-DEFAULT:     -lc
// CHECK-DEFAULT:     crtend.o
// CHECK-DEFAULT:     crtn.o
// CHECK-DEFAULT-NOT: values-Xc.o
// CHECK-DEFAULT-NOT: values-xpg4.o

// RUN: %clang -ansi -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ANSI %s
// CHECK-ANSI-NOT: warning:
// CHECK-ANSI:     values-Xc.o
// CHECK-ANSI:     values-xpg6.o
// CHECK-ANSI-NOT: values-Xa.o
// CHECK-ANSI-NOT: values-xpg4.o

// RUN: %clang -std=gnu89 -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GNU89 %s
// CHECK-GNU89-NOT: warning:
// CHECK-GNU89:     values-Xa.o
// CHECK-GNU89:     values-xpg4.o
// CHECK-GNU89-NOT: values-Xc.o
// CHECK-GNU89-NOT: values-xpg6.o

// -nostartfiles suppresses crt/values files but keeps default libraries.
// RUN: %clang -nostartfiles -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-NOSTARTFILES %s
// CHECK-NOSTARTFILES-NOT: warning:
// CHECK-NOSTARTFILES:     "-lgcc"
// CHECK-NOSTARTFILES:     "-lc"
// CHECK-NOSTARTFILES-NOT: /crt{{[^.]+}}.o
// CHECK-NOSTARTFILES-NOT: /values-{{[^.]+}}.o

// -nodefaultlibs suppresses libraries but keeps crt/values files.
// RUN: %clang -nodefaultlibs -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-NODEFAULTLIBS %s
// CHECK-NODEFAULTLIBS-NOT: warning:
// CHECK-NODEFAULTLIBS:     crt1.o
// CHECK-NODEFAULTLIBS:     values-Xa.o
// CHECK-NODEFAULTLIBS:     crtend.o
// CHECK-NODEFAULTLIBS-NOT: "-lgcc"
// CHECK-NODEFAULTLIBS-NOT: "-lc"

// -r suppresses crt/values files, entry point, and libraries like -nostdlib.
// RUN: %clang -r -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-RELOCATABLE %s
// CHECK-RELOCATABLE-NOT: warning:
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE-NOT: "-e"
// CHECK-RELOCATABLE-NOT: "-l
// CHECK-RELOCATABLE-NOT: /crt{{[^.]+}}.o
// CHECK-RELOCATABLE-NOT: /values-{{[^.]+}}.o

// -fuse-ld=gld selects GNU ld: different path, no -C, adds emulation and
// --eh-frame-hdr, uses --as-needed/--no-as-needed instead of -z ignore/-z record.
// RUN: %clang -fuse-ld=gld -### %s 2>&1 \
// RUN:     --target=x86_64-pc-illumos \
// RUN:     --sysroot=%S/Inputs/illumos_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GLD %s
// CHECK-GLD-NOT: warning:
// CHECK-GLD:     "/usr/gnu/bin/ld"
// CHECK-GLD:     "-m" "elf_x86_64_sol2"
// CHECK-GLD:     "--eh-frame-hdr"
// CHECK-GLD:     "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-GLD-NOT: "-C"
// CHECK-GLD-NOT: "-z" "ignore"
// CHECK-GLD-NOT: "-z" "record"
