// REQUIRES: aarch64-registered-target

// RUN: rm -f %t*

// -O1, no tagging: both are unsafe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -S -emit-llvm -c %s -o - | FileCheck %s
// RUN: %clang    -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -S -emit-llvm -c %s -o - | FileCheck %s

// Full LTO: both are unsafe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -c %s -flto=full -o %t.lto1.bc
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -c -DBUILD2 %s -flto=full -o %t.lto2.bc
// RUN: llvm-lto2 run -o %t.lto %t.lto1.bc %t.lto2.bc -save-temps -O1 \
// RUN:  -r %t.lto1.bc,fn,plx \
// RUN:  -r %t.lto1.bc,use,lx \
// RUN:  -r %t.lto1.bc,use_local,plx \
// RUN:  -r %t.lto1.bc,w, \
// RUN:  -r %t.lto2.bc,use,plx \
// RUN:  -r %t.lto2.bc,z,
// RUN: llvm-dis %t.lto.0.5.precodegen.bc -o - | FileCheck %s

// Full LTO, new PM: both are unsafe.
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -c %s -flto=full -o %t.ltonewpm1.bc
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -c -DBUILD2 %s -flto=full -o %t.ltonewpm2.bc
// RUN: llvm-lto2 run -use-new-pm -o %t.ltonewpm %t.ltonewpm1.bc %t.ltonewpm2.bc -save-temps -O1 \
// RUN:  -r %t.ltonewpm1.bc,fn,plx \
// RUN:  -r %t.ltonewpm1.bc,use,lx \
// RUN:  -r %t.ltonewpm1.bc,use_local,plx \
// RUN:  -r %t.ltonewpm1.bc,w, \
// RUN:  -r %t.ltonewpm2.bc,use,plx \
// RUN:  -r %t.ltonewpm2.bc,z,
// RUN: llvm-dis %t.ltonewpm.0.5.precodegen.bc -o - | FileCheck %s

// Thin LTO: both are unsafe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -c %s -flto=thin -o %t.thinlto1.bc
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -c -DBUILD2 %s -flto=thin -o %t.thinlto2.bc
// RUN: llvm-lto2 run -o %t.thinlto %t.thinlto1.bc %t.thinlto2.bc -save-temps -O1 \
// RUN:  -r %t.thinlto1.bc,fn,plx \
// RUN:  -r %t.thinlto1.bc,use,lx \
// RUN:  -r %t.thinlto1.bc,use_local,plx \
// RUN:  -r %t.thinlto1.bc,w, \
// RUN:  -r %t.thinlto2.bc,use,plx \
// RUN:  -r %t.thinlto2.bc,z,
// RUN: llvm-dis %t.thinlto.1.5.precodegen.bc -o - | FileCheck %s

// Thin LTO, new PM: both are unsafe.
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -c %s -flto=thin -o %t.thinltonewpm1.bc
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -c -DBUILD2 %s -flto=thin -o %t.thinltonewpm2.bc
// RUN: llvm-lto2 run -use-new-pm -o %t.thinltonewpm %t.thinltonewpm1.bc %t.thinltonewpm2.bc -save-temps -O1 \
// RUN:  -r %t.thinltonewpm1.bc,fn,plx \
// RUN:  -r %t.thinltonewpm1.bc,use,lx \
// RUN:  -r %t.thinltonewpm1.bc,use_local,plx \
// RUN:  -r %t.thinltonewpm1.bc,w, \
// RUN:  -r %t.thinltonewpm2.bc,use,plx \
// RUN:  -r %t.thinltonewpm2.bc,z,
// RUN: llvm-dis %t.thinltonewpm.1.5.precodegen.bc -o - | FileCheck %s

// Now with MTE.
// RUN: rm -f %t*

// -O0: both are unsafe.
// RUN: %clang -fno-experimental-new-pass-manager -O0 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -S -emit-llvm -c %s -o - | FileCheck %s
// RUN: %clang    -fexperimental-new-pass-manager -O0 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -S -emit-llvm -c %s -o - | FileCheck %s

// No LTO: just one is safe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -S -emit-llvm -c %s -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE
// RUN: %clang    -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -S -emit-llvm -c %s -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE

// FIXME: Full LTO: both are safe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c %s -flto=full -o %t.lto1.bc
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c -DBUILD2 %s -flto=full -o %t.lto2.bc
// RUN: llvm-lto2 run -o %t.lto %t.lto1.bc %t.lto2.bc -save-temps -O1 \
// RUN:  -r %t.lto1.bc,fn,plx \
// RUN:  -r %t.lto1.bc,use,lx \
// RUN:  -r %t.lto1.bc,use_local,plx \
// RUN:  -r %t.lto1.bc,w, \
// RUN:  -r %t.lto2.bc,use,plx \
// RUN:  -r %t.lto2.bc,z,
// RUN: llvm-dis %t.lto.0.5.precodegen.bc -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE

// FIXME: Full LTO, new PM: both are safe.
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c %s -flto=full -o %t.ltonewpm1.bc
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c -DBUILD2 %s -flto=full -o %t.ltonewpm2.bc
// RUN: llvm-lto2 run -use-new-pm -o %t.ltonewpm %t.ltonewpm1.bc %t.ltonewpm2.bc -save-temps -O1 \
// RUN:  -r %t.ltonewpm1.bc,fn,plx \
// RUN:  -r %t.ltonewpm1.bc,use,lx \
// RUN:  -r %t.ltonewpm1.bc,use_local,plx \
// RUN:  -r %t.ltonewpm1.bc,w, \
// RUN:  -r %t.ltonewpm2.bc,use,plx \
// RUN:  -r %t.ltonewpm2.bc,z,
// RUN: llvm-dis %t.ltonewpm.0.5.precodegen.bc -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE

// FIXME: Thin LTO: both are safe.
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c %s -flto=thin -o %t.thinlto1.bc
// RUN: %clang -fno-experimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c -DBUILD2 %s -flto=thin -o %t.thinlto2.bc
// RUN: llvm-lto2 run -o %t.thinlto %t.thinlto1.bc %t.thinlto2.bc -save-temps -O1 \
// RUN:  -r %t.thinlto1.bc,fn,plx \
// RUN:  -r %t.thinlto1.bc,use,lx \
// RUN:  -r %t.thinlto1.bc,use_local,plx \
// RUN:  -r %t.thinlto1.bc,w, \
// RUN:  -r %t.thinlto2.bc,use,plx \
// RUN:  -r %t.thinlto2.bc,z,
// RUN: llvm-dis %t.thinlto.1.5.precodegen.bc -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE

// FIXME: Thin LTO, new PM: both are safe.
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c %s -flto=thin -o %t.thinltonewpm1.bc
// RUN: %clang -fexperimental-new-pass-manager -O1 -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag -c -DBUILD2 %s -flto=thin -o %t.thinltonewpm2.bc
// RUN: llvm-lto2 run -use-new-pm -o %t.thinltonewpm %t.thinltonewpm1.bc %t.thinltonewpm2.bc -save-temps -O1 \
// RUN:  -r %t.thinltonewpm1.bc,fn,plx \
// RUN:  -r %t.thinltonewpm1.bc,use,lx \
// RUN:  -r %t.thinltonewpm1.bc,use_local,plx \
// RUN:  -r %t.thinltonewpm1.bc,w, \
// RUN:  -r %t.thinltonewpm2.bc,use,plx \
// RUN:  -r %t.thinltonewpm2.bc,z,
// RUN: llvm-dis %t.thinltonewpm.1.5.precodegen.bc -o - | FileCheck %s -check-prefixes=XUNSAFE,YSAFE

void use(int *p);

#ifdef BUILD2

int z;
__attribute__((noinline)) void use(int *p) { *p = z; }

#else

char w;
__attribute__((noinline)) void use_local(char *p) { *p = w; }

__attribute__((visibility("default"))) int fn() {
  // XUNSAFE: alloca i32, align 4{{$}}
  // XSAFE: alloca i32, align 4, !stack-safe
  int x;
  use(&x);

  // YUNSAFE-NEXT: alloca i8, align 4{{$}}
  // YSAFE-NEXT: alloca i8, align 4, !stack-safe
  char y;
  use_local(&y);
  return x + y;
}

// CHECK-NOT: !stack-safe

#endif
