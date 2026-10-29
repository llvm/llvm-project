// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-O0.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-O0.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct NotTrivial {
  char buf[32];
  NotTrivial(const NotTrivial &);
  ~NotTrivial();
};

struct Base1 {
  virtual ~Base1();
  virtual int other();
};

struct Base2 {
  virtual void take(const NotTrivial desc);
};

struct Derived : Base1, Base2 {
  void take(const NotTrivial desc) override;
};

void Derived::take(const NotTrivial desc) {}

// No copy of the parameter, at either optimization level.

// CIR-LABEL: cir.func{{.*}} @_ZThn8_N7Derived4takeE10NotTrivial
// CIR-SAME:    %{{[^ :]+}}: !cir.ptr<!rec_Derived> {llvm.noundef}
// CIR-SAME:    %[[DESC:[^ :]+]]: !cir.ptr<!rec_NotTrivial> {llvm.align = 1 : i64, llvm.dereferenceable = 32 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NOT:     cir.alloca "desc"
// CIR:         cir.call @_ZN7Derived4takeE10NotTrivial(%{{[^,)]+}}, %[[DESC]]) : (!cir.ptr<!rec_Derived> {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_NotTrivial> {llvm.align = 1 : i64, llvm.dereferenceable = 32 : i64, llvm.nofreeobj, llvm.noundef}) -> ()

// The overrider takes the same pointer without byval, which is the definition
// side of the same contract.

// LLVM-LABEL: define dso_local void @_ZN7Derived4takeE10NotTrivial(
// LLVM-SAME:    ptr noundef nonnull align 8 dereferenceable(16) %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(32) %{{[^,)]+}})

// LLVM-LABEL: define dso_local void @_ZThn8_N7Derived4takeE10NotTrivial(
// LLVM-SAME:    ptr noundef %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(32) %[[DESC:[^,)]+]])
// LLVM-CIR:     call void @_ZN7Derived4takeE10NotTrivial(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(32) %[[DESC]])
// OGCG:         tail call void @_ZN7Derived4takeE10NotTrivial(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(32) %[[DESC]])
