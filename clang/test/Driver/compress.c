// RUN: %clang -### -c -integrated-as -Wa,-compress-debug-sections %s 2>&1 | FileCheck --check-prefix=COMPRESS_DEBUG %s
// RUN: %clang -### -c -integrated-as -Wa,--compress-debug-sections %s 2>&1 | FileCheck --check-prefix=COMPRESS_DEBUG %s
// REQUIRES: zlib

// COMPRESS_DEBUG: "-compress-debug-sections"

// RUN: %clang -### -c -integrated-as -Wa,--compress-debug-sections -Wa,--nocompress-debug-sections %s 2>&1 | FileCheck --check-prefix=NOCOMPRESS_DEBUG %s
// NOCOMPRESS_DEBUG-NOT: "-compress-debug-sections"
