// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O1 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O3 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-ASAN

// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-ASAN

// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O1 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O3 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang     -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KASAN

// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target i386-unknown-linux -fsanitize=kernel-address %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KASAN

// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O1 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O3 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-HWASAN

// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-HWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-HWASAN

// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O1 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O3 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KHWASAN

// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -fsanitize=kernel-hwaddress %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefix=CHECK-KHWASAN
// Verify that -fsanitize={address,hwaddres,kernel-address,kernel-hwaddress} invokes ASan, HWAsan, KASan or KHWASan instrumentation.

int foo(int *a) { return *a; }
// CHECK-ASAN: __asan_init
// CHECK-KASAN: __asan_{{.*}}load4_noabort
// CHECK-HWASAN: __hwasan_init
// CHECK-KHWASAN: __hwasan_tls
