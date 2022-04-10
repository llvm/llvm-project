// REQUIRES: zlib

// RUN: %clang -### -fintegrated-as -Wa,-compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-_COMPRESS_DEBUG_SECTIONS %s
// CHECK-_COMPRESS_DEBUG_SECTIONS: "-compress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-__COMPRESS_DEBUG_SECTIONS %s
// CHECK-__COMPRESS_DEBUG_SECTIONS: "--compress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-POSNEG %s
// CHECK-POSNEG: "--compress-debug-sections"
// CHECK-POSNEG: "--nocompress-debug-sections"

// RUN: %clang -### -fintegrated-as -Wa,-compress-debug-sections -Wa,--compress-debug-sections -c %s 2>&1 | FileCheck -check-prefix CHECK-MULTIPLE %s
// CHECK-MULTIPLE: "-compress-debug-sections"
// CHECK-MULTIPLE: "--compress-debug-sections"

// RUN: %clang -### -fintegrated-as -gz -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ %s
// RUN: %clang -### -fintegrated-as -gz -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ %s
// CHECK-OPT_GZ: "--compress-debug-sections=zlib"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -gz=none -x assembler %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -gz=none %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_NONE %s
// CHECK-OPT_GZ_EQ_NONE: {{.* "-cc1(as)?".* "--compress-debug-sections=none"}}
// CHECK-OPT_GZ_EQ_NONE: "--compress-debug-sections=none"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -gz=zlib -x assembler %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// RUN: %clang -### -target x86_64-unknown-linux-gnu -gz=zlib %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// RUN: %clang -### -target x86_64-unknown-freebsd -gz=zlib %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_ZLIB %s
// CHECK-OPT_GZ_EQ_ZLIB: {{.* "-cc1(as)?".* "--compress-debug-sections=zlib"}}
// CHECK-OPT_GZ_EQ_ZLIB: "--compress-debug-sections=zlib"

// RUN: %clang -### -fintegrated-as -gz=invalid -x assembler -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_INVALID %s
// RUN: %clang -### -fintegrated-as -gz=invalid -c %s 2>&1 | FileCheck -check-prefix CHECK-OPT_GZ_EQ_INVALID %s
// CHECK-OPT_GZ_EQ_INVALID: error: unsupported argument 'invalid' to option '-gz='
