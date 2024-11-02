// REQUIRES: zlib
// REQUIRES: x86-registered-target

// RUN: %clang -cc1as -triple i686 --compress-debug-sections %s -o /dev/null
// RUN: %clang -cc1as -triple i686 -compress-debug-sections=zlib %s -o /dev/null

// RUN: %if zstd %{ %clang -cc1as -triple x86_64 -filetype obj -compress-debug-sections=zstd %s -o %t %}
// RUN: %if zstd %{ llvm-readelf -S -x .debug_str %t | FileCheck %s --check-prefix=ZSTD %}

// ZSTD: .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MSC 0 0  8
// ZSTD: Hex dump of section '.debug_str':
// ZSTD: 0000 02000000 00000000

.section        .debug_str,"MS",@progbits,1
.asciz  "perfectly compressable data sample *****************************************"
