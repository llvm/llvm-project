// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

signed char sc;
unsigned char uc;
signed short ss;
unsigned short us;
signed int si;
unsigned int ui;
signed long long sll;
unsigned long long ull;
float f1, f2;
double d1, d2;

void test_nontemporal_store() {
// CIR-LABEL: cir.func {{.*}}@_Z22test_nontemporal_storev
// CIR: cir.store nontemporal align(1) {{%.*}}, {{%.*}} : !u8i, !cir.ptr<!u8i>
// CIR: cir.store nontemporal align(1) {{%.*}}, {{%.*}} : !u8i, !cir.ptr<!u8i>
// CIR: cir.store nontemporal align(1) {{%.*}}, {{%.*}} : !s8i, !cir.ptr<!s8i>
// CIR: cir.store nontemporal align(2) {{%.*}}, {{%.*}} : !u16i, !cir.ptr<!u16i>
// CIR: cir.store nontemporal align(4) {{%.*}}, {{%.*}} : !s32i, !cir.ptr<!s32i>
// CIR: cir.store nontemporal align(8) {{%.*}}, {{%.*}} : !u64i, !cir.ptr<!u64i>
// CIR: cir.store nontemporal align(4) {{%.*}}, {{%.*}} : !cir.float, !cir.ptr<!cir.float>
// CIR: cir.store nontemporal align(8) {{%.*}}, {{%.*}} : !cir.double, !cir.ptr<!cir.double>
// CIR: cir.return

// LLVM-LABEL: define dso_local void @_Z22test_nontemporal_storev
// LLVM: store i8 1, ptr @uc, align 1, !nontemporal
// LLVM: store i8 1, ptr @uc, align 1, !nontemporal
// LLVM: store i8 1, ptr @sc, align 1, !nontemporal
// LLVM: store i16 1, ptr @us, align 2, !nontemporal
// LLVM: store i32 1, ptr @si, align 4, !nontemporal
// LLVM: store i64 1, ptr @ull, align 8, !nontemporal
// LLVM: store float 1.0{{.*}}, ptr @f1, align 4, !nontemporal
// LLVM: store double 1.0{{.*}}, ptr @d1, align 8, !nontemporal
// LLVM: ret void

  __builtin_nontemporal_store(true, &uc);
  __builtin_nontemporal_store(1, &uc);
  __builtin_nontemporal_store(1, &sc);
  __builtin_nontemporal_store(1, &us);
  __builtin_nontemporal_store(1, &si);
  __builtin_nontemporal_store(1, &ull);
  __builtin_nontemporal_store(1.0, &f1);
  __builtin_nontemporal_store(1.0, &d1);
}

void test_nontemporal_load() {
// CIR-LABEL: cir.func {{.*}}@_Z21test_nontemporal_loadv
// CIR: cir.load nontemporal align(1) {{%.*}} : !cir.ptr<!s8i>, !s8i
// CIR: cir.load nontemporal align(1) {{%.*}} : !cir.ptr<!u8i>, !u8i
// CIR: cir.load nontemporal align(2) {{%.*}} : !cir.ptr<!s16i>, !s16i
// CIR: cir.load nontemporal align(4) {{%.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.load nontemporal align(8) {{%.*}} : !cir.ptr<!s64i>, !s64i
// CIR: cir.load nontemporal align(4) {{%.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: cir.load nontemporal align(8) {{%.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: cir.return

// LLVM-LABEL: define dso_local void @_Z21test_nontemporal_loadv
// LLVM: load i8, ptr @sc, align 1, !nontemporal
// LLVM: load i8, ptr @uc, align 1, !nontemporal
// LLVM: load i16, ptr @ss, align 2, !nontemporal
// LLVM: load i32, ptr @ui, align 4, !nontemporal
// LLVM: load i64, ptr @sll, align 8, !nontemporal
// LLVM: load float, ptr @f2, align 4, !nontemporal
// LLVM: load double, ptr @d2, align 8, !nontemporal
// LLVM: ret void

  uc = __builtin_nontemporal_load(&sc);
  sc = __builtin_nontemporal_load(&uc);
  us = __builtin_nontemporal_load(&ss);
  si = __builtin_nontemporal_load(&ui);
  ull = __builtin_nontemporal_load(&sll);
  f1 = __builtin_nontemporal_load(&f2);
  d1 = __builtin_nontemporal_load(&d2);
}
