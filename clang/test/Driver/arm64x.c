// RUN: %clang -c -marm64x  --target=arm64ec-pc-windows-msvc -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -marm64x  --target=arm64ec-pc-windows-gnu -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -marm64x  --target=aarch64-pc-windows-msvc -### %s 2>&1 | FileCheck %s
// RUN: %clang -c -marm64x  --target=aarch64-pc-windows-gnu -### %s 2>&1 | FileCheck %s

// CHECK:      "-cc1" "-triple" "aarch64-pc-windows-{{.*}}" "-emit-obj"
// CHECK-NEXT: "-cc1" "-triple" "arm64ec-pc-windows-{{.*}}" "-emit-obj"
// CHECK-NEXT: llvm-objcopy" "--add-section=.obj.arm64ec={{.*}}arm64x-arm64ec-{{.*}}.o" "--set-section-flags=.obj.arm64ec=exclude" "{{.*}}arm64x-aarch64-{{.*}}.o" "arm64x.o"
