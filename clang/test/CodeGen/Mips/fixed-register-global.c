// REQUIRES: mips-registered-target
// RUN: %clang --target=mipsel-unknown-netbsd -ffixed-24 -O2 -S %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang --target=mips64el-unknown-netbsd -ffixed-24 -O2 -S %s -o - \
// RUN:   | FileCheck %s

// Match GCC's global register variable syntax used by the NetBSD MIPS kernel.
register void *mips_curlwp asm("$24");

// CHECK-LABEL: get_curlwp:
// CHECK: move $2, $24
void *get_curlwp(void) {
  return mips_curlwp;
}

// CHECK-LABEL: set_curlwp:
// CHECK: move $24, $4
void set_curlwp(void *lwp) {
  mips_curlwp = lwp;
}
