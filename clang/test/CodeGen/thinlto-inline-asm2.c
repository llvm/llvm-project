// REQUIRES: x86-registered-target

// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -flto=thin -emit-llvm-bc %t/a.c -o %t/a.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -flto=thin -emit-llvm-bc %t/b.c -o %t/b.bc
// RUN: llvm-nm %t/a.bc | FileCheck %s --check-prefix=NM

// RUN: llvm-lto2 run -lto-opaque-pointers %t/a.bc %t/b.bc -o %t/out -save-temps -r=%t/a.bc,ref,plx -r=%t/b.bc,ff_h264_cabac_tables,pl
// RUN: llvm-dis < %t/out.2.2.internalize.bc | FileCheck %s

//--- a.c
/// IR symtab does not track inline asm symbols, so we don't know
/// ff_h264_cabac_tables is undefined.
// NM-NOT: {{.}}
// NM:     ---------------- T ref
// NM-NOT: {{.}}
const char *ref(void) {
  const char *ret;
  asm("lea ff_h264_cabac_tables(%%rip), %0" : "=r"(ret));
  return ret;
}

//--- b.c
/// ff_h264_cabac_tables has __attribute__((used)) in the source code, which means
/// its definition must be retained because there can be references the compiler
/// cannot see (inline asm reference). Test we don't internalize it.
// CHECK: @ff_h264_cabac_tables = dso_local constant [1 x i8] c"\09"
__attribute__((used))
const char ff_h264_cabac_tables[1] = "\x09";
