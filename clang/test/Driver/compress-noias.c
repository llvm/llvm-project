// REQUIRES: zlib
// REQUIRES: x86-registered-target

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -Wa,-compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-_COMPRESS_DEBUG_SECTIONS %s
// CHECK-_COMPRESS_DEBUG_SECTIONS: "-compress-debug-sections"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-__COMPRESS_DEBUG_SECTIONS %s
// CHECK-__COMPRESS_DEBUG_SECTIONS: "--compress-debug-sections"

// RUN: %clang -### --target=i386-pc-solaris2.11 -fno-integrated-as -Wa,-compress-debug-sections=zlib -c %s 2>&1 | FileCheck -check-prefix CHECK-_COMPRESS_DEBUG_SECTIONS-ZLIB %s
// CHECK-_COMPRESS_DEBUG_SECTIONS-ZLIB: "-compress-debug-sections=zlib"

// RUN: %clang -### --target=i386-pc-solaris2.11 -fno-integrated-as -Wa,--compress-debug-sections=zlib -c %s 2>&1 | FileCheck -check-prefix CHECK-__COMPRESS_DEBUG_SECTIONS-ZLIB %s
// CHECK-__COMPRESS_DEBUG_SECTIONS-ZLIB: "--compress-debug-sections=zlib"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-POSNEG %s
// CHECK-POSNEG: "--compress-debug-sections"
// CHECK-POSNEG: "--nocompress-debug-sections"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -Wa,-compress-debug-sections -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-MULTIPLE %s
// CHECK-MULTIPLE: "-compress-debug-sections"
// CHECK-MULTIPLE: "--compress-debug-sections"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ %s
// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ %s
// CHECK-OPT_GZ: "--compress-debug-sections=zlib"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=none -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=none -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// CHECK-OPT_GZ_EQ_NONE: "--compress-debug-sections=none"

// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=zlib -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// RUN: %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=zlib -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// CHECK-OPT_GZ_EQ_ZLIB: "--compress-debug-sections=zlib"

// RUN: not %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=invalid -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_INVALID %s
// RUN: not %clang -### -target i686-unknown-linux-gnu -fno-integrated-as -gz=invalid -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_INVALID %s
// CHECK-OPT_GZ_EQ_INVALID: error: unsupported argument 'invalid' to option '-gz='

