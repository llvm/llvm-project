// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple aarch64-none-elf \
// RUN:   -O2 \
// RUN:   -emit-llvm -fexperimental-max-bitint-width=1024 -o - %s | FileCheck %s

_BitInt(129) v = -1;
int h(_BitInt(129));

// CHECK: declare i32 @h(ptr noundef)
int largerthan128() {
   return h(v);
}


