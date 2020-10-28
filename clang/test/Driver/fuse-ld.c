/// The absolute path warning is enabled by -Wfuse-ld-path and -Wextra.
// RUN: %clang %s -### -target x86_64-unknown-linux -Wfuse-ld-path \
// RUN:   -fuse-ld=/usr/local/bin/or1k-linux-ld 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ABSOLUTE-LD
// CHECK-ABSOLUTE-LD: warning: '-fuse-ld=' taking a path is deprecated. Use '--ld-path=' instead
// CHECK-ABSOLUTE-LD: /usr/local/bin/or1k-linux-ld

// RUN: %clang %s -### -target x86_64-unknown-linux -Wextra \
// RUN:   -fuse-ld=/usr/local/bin/or1k-linux-ld 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ABSOLUTE-LD

// RUN: %clang %s -### -target x86_64-unknown-linux \
// RUN:   -fuse-ld=/usr/local/bin/or1k-linux-ld 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-WARN
// CHECK-NO-WARN-NOT: warning:

// RUN: %clang %s -### \
// RUN:     -target x86_64-unknown-freebsd 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FREEBSD-LD
// CHECK-FREEBSD-LD: ld

// RUN: %clang %s -### -fuse-ld=bfd \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:     -target x86_64-unknown-freebsd \
// RUN:     -B%S/Inputs/basic_freebsd_tree/usr/bin 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-FREEBSD-BFD
// CHECK-FREEBSD-BFD: Inputs/basic_freebsd_tree/usr/bin{{/|\\+}}ld.bfd

// RUN: %clang %s -### -fuse-ld=gold \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:     -target x86_64-unknown-freebsd \
// RUN:     -B%S/Inputs/basic_freebsd_tree/usr/bin 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-FREEBSD-GOLD
// CHECK-FREEBSD-GOLD: Inputs/basic_freebsd_tree/usr/bin{{/|\\+}}ld.gold

// RUN: %clang %s -### -fuse-ld=plib \
// RUN:     --sysroot=%S/Inputs/basic_freebsd_tree \
// RUN:     -target x86_64-unknown-freebsd \
// RUN:     -B%S/Inputs/basic_freebsd_tree/usr/bin 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-FREEBSD-PLIB
// CHECK-FREEBSD-PLIB: error: invalid linker name

// RUN: %clang %s -### -fuse-ld=ld \
// RUN:     -target arm-linux-androideabi \
// RUN:     -B%S/Inputs/basic_android_tree/bin/arm-linux-androideabi- 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ANDROID-ARM-LD
// CHECK-ANDROID-ARM-LD: Inputs/basic_android_tree/bin{{/|\\+}}arm-linux-androideabi-ld

// RUN: %clang %s -### -fuse-ld=bfd \
// RUN:     -target arm-linux-androideabi \
// RUN:     -B%S/Inputs/basic_android_tree/bin/arm-linux-androideabi- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ANDROID-ARM-BFD
// CHECK-ANDROID-ARM-BFD: Inputs/basic_android_tree/bin{{/|\\+}}arm-linux-androideabi-ld.bfd

// RUN: %clang %s -### -fuse-ld=gold \
// RUN:     -target arm-linux-androideabi \
// RUN:     -B%S/Inputs/basic_android_tree/bin/arm-linux-androideabi- 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ANDROID-ARM-GOLD
// CHECK-ANDROID-ARM-GOLD: Inputs/basic_android_tree/bin{{/|\\+}}arm-linux-androideabi-ld.gold

// RUN: %clang %s -### -fuse-ld=ld \
// RUN:     -target arm-linux-androideabi \
// RUN:     -gcc-toolchain %S/Inputs/basic_android_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ANDROID-ARM-LD-TC
// CHECK-ANDROID-ARM-LD-TC: Inputs/basic_android_tree/lib/gcc/arm-linux-androideabi/4.4.3/../../../../arm-linux-androideabi/bin{{/|\\+}}ld

// RUN: %clang %s -### -fuse-ld=bfd \
// RUN:     -target arm-linux-androideabi \
// RUN:     -gcc-toolchain %S/Inputs/basic_android_tree 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ANDROID-ARM-BFD-TC
// CHECK-ANDROID-ARM-BFD-TC: Inputs/basic_android_tree/lib/gcc/arm-linux-androideabi/4.4.3/../../../../arm-linux-androideabi/bin{{/|\\+}}ld.bfd

// RUN: %clang %s -### -fuse-ld=gold \
// RUN:     -target arm-linux-androideabi \
// RUN:     -gcc-toolchain %S/Inputs/basic_android_tree 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ANDROID-ARM-GOLD-TC
// CHECK-ANDROID-ARM-GOLD-TC: Inputs/basic_android_tree/lib/gcc/arm-linux-androideabi/4.4.3/../../../../arm-linux-androideabi/bin{{/|\\+}}ld.gold


// RUN: %clang %s -### -fuse-ld=link \
// RUN:     -target i686-unknown-windows-msvc 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-WINDOWS-MSVC-LINK
// CHECK-WINDOWS-MSVC-LINK: "{{.*}}link.exe"
// CHECK-WINDOWS-MSVC-LINK-SAME: "-out:{{.*}}"

// RUN: %clang %s -### -fuse-ld=lld \
// RUN:     -target i686-unknown-windows-msvc 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-WINDOWS-MSVC-LLD
// CHECK-WINDOWS-MSVC-LLD: "{{.*}}lld-link{{\.exe"|"}}
// CHECK-WINDOWS-MSVC-LLD-SAME: "-out:{{.*}}"

// RUN: %clang %s -### -fuse-ld=lld-link \
// RUN:     -target i686-unknown-windows-msvc 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-WINDOWS-MSVC-LLD-LINK
// CHECK-WINDOWS-MSVC-LLD-LINK: "{{.*}}lld-link{{\.exe"|"}}
// CHECK-WINDOWS-MSVC-LLD-LINK-SAME: "-out:{{.*}}"

// RUN: %clang %s -### -fuse-ld=bfd \
// RUN:     -target i686-unknown-windows-msvc \
// RUN:     -B %S/Inputs/Windows/usr/bin 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK-WINDOWS-MSVC-BFD
// CHECK-WINDOWS-MSVC-BFD: "{{.*}}ld.bfd"
// CHECK-WINDOWS-MSVC-BFD-SAME: "-o"
