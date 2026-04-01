// RUN: %clang -c -marm64x  --target=arm64ec-pc-windows-msvc -### %s 2>&1 | FileCheck %s

// CHECK:      "-cc1" "-triple" "arm64ec-pc-windows-msvc{{.*}}" "-emit-obj"
// CHECK-NEXT: "-cc1" "-triple" "aarch64-pc-windows-msvc{{.*}}" "-emit-obj"
// CHECK-NEXT: llvm-ar" "rcsD" "--whole-archive" "arm64x.o" "{{.*}}arm64x-arm64ec-{{.*}}.o" "{{.*}}arm64x-aarch64-{{.*}}.o"
