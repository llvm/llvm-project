// REQUIRES: zstd

// RUN: %clang -### --target=aarch64-unknown-linux-gnu -gz=zstd -x assembler %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-pc-freebsd -gz=zstd %s 2>&1 | FileCheck %s

// CHECK: {{"-cc1(as)?".* "--compress-debug-sections=zstd"}}
// CHECK: "--compress-debug-sections=zstd"

// RUN: %clang -### -c -ftime-trace -ftime-trace-compress=zstd %s -o a.o 2>&1 | FileCheck %s --check-prefix=TIME-TRACE-ZSTD
// RUN: %clang -### -c -ftime-trace=a.json.zst %s -o a.o 2>&1 | FileCheck %s --check-prefix=TIME-TRACE-ZSTD
// RUN: %clang -### -c -ftime-trace=a.json.zst -ftime-trace-compress=none -ftime-trace-compress=infer %s -o a.o 2>&1 | FileCheck %s --check-prefix=TIME-TRACE-ZSTD
// TIME-TRACE-ZSTD: "-cc1"{{.*}} "-ftime-trace=a.json.zst" "-ftime-trace-compress=zstd"
