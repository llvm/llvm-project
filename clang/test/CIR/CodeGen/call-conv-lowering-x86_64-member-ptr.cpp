// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct S {
  void m();
  int f;
};

// CXXABILowering lowers a pointer to member function to an anonymous record of
// two eightbytes, whose coerce record is that same type.  Equal types still
// need the rewrite, because the coercion is what flattens the record into the
// two registers it passes in.
// CIR: ![[MEMPTR:rec_anon_struct[0-9]*]] = !cir.struct<{data !s64i, data !s64i}>
void take_method_ptr(void (S::*p)()) { (void)p; }

// CIR: cir.func {{.*}}@_Z15take_method_ptrM1SFvvE(%arg0: !s64i{{.*}}, %arg1: !s64i{{.*}})
// LLVM: define dso_local void @_Z15take_method_ptrM1SFvvE(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// A pointer to data member is a single eightbyte, so it needs no coercion at
// all and is the control for the case above.
void take_data_ptr(int S::*p) { (void)p; }

// CIR: cir.func {{.*}}@_Z13take_data_ptrM1Si(%arg0: !s64i{{.*}})
// LLVM: define dso_local void @_Z13take_data_ptrM1Si(i64 %{{[^,)]+}})
