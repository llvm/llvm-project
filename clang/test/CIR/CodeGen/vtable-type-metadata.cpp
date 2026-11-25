// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -flto-unit -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -flto-unit -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -flto-unit -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-nolto.cir
// RUN: FileCheck --check-prefix=NOLTO --input-file=%t-nolto.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fdevirtualize-speculatively -fclangir -emit-llvm %s -o %t-spec-cir.ll
// RUN: FileCheck --check-prefix=SPEC --input-file=%t-spec-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fdevirtualize-speculatively -emit-llvm %s -o %t-spec.ll
// RUN: FileCheck --check-prefix=SPEC --input-file=%t-spec.ll %s

// NOLTO-NOT: type_metadata

struct A {
  virtual int f();
};
int A::f() { return 1; }

struct B {
  virtual int g();
};

struct C : A, B {
  int f() override;
  int g() override;
};
int C::f() { return 2; }
int C::g() { return 3; }

namespace {
struct D : A {
  int f() override { return 4; }
};
struct E : D {
  int f() override { return 5; }
};
} // namespace

A *makeE() { return new E; }

// Key function in another TU: available_externally vtable.
struct F {
  virtual int h();
  F() {}
};
F *makeF() { return new F; }

// CIR: cir.global {{.*}} @_ZTV1A = {{.*}} type_metadata([#cir.type_metadata<16, "_ZTS1A">, #cir.type_metadata<16, "_ZTSM1AFivE.virtual">])
// CIR: cir.global {{.*}} @_ZTV1C = {{.*}} type_metadata([#cir.type_metadata<16, "_ZTS1A">, #cir.type_metadata<16, "_ZTSM1AFivE.virtual">, #cir.type_metadata<24, "_ZTSM1AFivE.virtual">, #cir.type_metadata<48, "_ZTSM1AFivE.virtual">, #cir.type_metadata<48, "_ZTS1B">, #cir.type_metadata<16, "_ZTSM1BFivE.virtual">, #cir.type_metadata<24, "_ZTSM1BFivE.virtual">, #cir.type_metadata<48, "_ZTSM1BFivE.virtual">, #cir.type_metadata<16, "_ZTS1C">, #cir.type_metadata<16, "_ZTSM1CFivE.virtual">, #cir.type_metadata<24, "_ZTSM1CFivE.virtual">, #cir.type_metadata<48, "_ZTSM1CFivE.virtual">])
// CIR: cir.global {{.*}} @_ZTVN12_GLOBAL__N_11EE = {{.*}} type_metadata([#cir.type_metadata<16, "_ZTS1A">, #cir.type_metadata<16, "_ZTSM1AFivE.virtual">, #cir.type_metadata<16, "_ZTSN12_GLOBAL__N_11DE", internal true>, #cir.type_metadata<16, "_ZTSMN12_GLOBAL__N_11DEFivE.virtual", internal true>, #cir.type_metadata<16, "_ZTSN12_GLOBAL__N_11EE", internal true>, #cir.type_metadata<16, "_ZTSMN12_GLOBAL__N_11EEFivE.virtual", internal true>])
// CIR: cir.global {{.*}} @_ZTVN12_GLOBAL__N_11DE = {{.*}} type_metadata([#cir.type_metadata<16, "_ZTS1A">, #cir.type_metadata<16, "_ZTSM1AFivE.virtual">, #cir.type_metadata<16, "_ZTSN12_GLOBAL__N_11DE", internal true>, #cir.type_metadata<16, "_ZTSMN12_GLOBAL__N_11DEFivE.virtual", internal true>])

// LLVM: @_ZTV1A = {{.*}}, !type ![[A16:[0-9]+]], !type ![[AF16:[0-9]+]]
// LLVM: @_ZTV1C = {{.*}}, !type ![[A16]], !type ![[AF16]], !type ![[AF24:[0-9]+]], !type ![[AF48:[0-9]+]], !type ![[B48:[0-9]+]], !type ![[BF16:[0-9]+]], !type ![[BF24:[0-9]+]], !type ![[BF48:[0-9]+]], !type ![[C16:[0-9]+]], !type ![[CF16:[0-9]+]], !type ![[CF24:[0-9]+]], !type ![[CF48:[0-9]+]]
// LLVM: @_ZTVN12_GLOBAL__N_11EE = {{.*}}, !type ![[A16]], !type ![[AF16]], !type ![[D16:[0-9]+]], !type ![[DF16:[0-9]+]], !type ![[E16:[0-9]+]], !type ![[EF16:[0-9]+]]
// LLVM: @_ZTVN12_GLOBAL__N_11DE = {{.*}}, !type ![[A16]], !type ![[AF16]], !type ![[D16]], !type ![[DF16]]

// LLVM: ![[A16]] = !{i64 16, !"_ZTS1A"}
// LLVM: ![[AF16]] = !{i64 16, !"_ZTSM1AFivE.virtual"}
// LLVM: ![[AF24]] = !{i64 24, !"_ZTSM1AFivE.virtual"}
// LLVM: ![[AF48]] = !{i64 48, !"_ZTSM1AFivE.virtual"}
// LLVM: ![[B48]] = !{i64 48, !"_ZTS1B"}
// LLVM: ![[BF16]] = !{i64 16, !"_ZTSM1BFivE.virtual"}
// LLVM: ![[BF24]] = !{i64 24, !"_ZTSM1BFivE.virtual"}
// LLVM: ![[BF48]] = !{i64 48, !"_ZTSM1BFivE.virtual"}
// LLVM: ![[C16]] = !{i64 16, !"_ZTS1C"}
// LLVM: ![[CF16]] = !{i64 16, !"_ZTSM1CFivE.virtual"}
// LLVM: ![[CF24]] = !{i64 24, !"_ZTSM1CFivE.virtual"}
// LLVM: ![[CF48]] = !{i64 48, !"_ZTSM1CFivE.virtual"}
// LLVM: ![[D16]] = !{i64 16, ![[D:[0-9]+]]}
// LLVM: ![[D]] = distinct !{}
// LLVM: ![[DF16]] = !{i64 16, ![[DF:[0-9]+]]}
// LLVM: ![[DF]] = distinct !{}
// LLVM: ![[E16]] = !{i64 16, ![[E:[0-9]+]]}
// LLVM: ![[E]] = distinct !{}
// LLVM: ![[EF16]] = !{i64 16, ![[EF:[0-9]+]]}
// LLVM: ![[EF]] = distinct !{}

// SPEC: @_ZTV1F = available_externally {{.*}}, !type ![[F16:[0-9]+]], !type ![[FF16:[0-9]+]]
// SPEC: @llvm.compiler.used = appending global [1 x ptr] [ptr @_ZTV1F], section "llvm.metadata"
// SPEC: ![[F16]] = !{i64 16, !"_ZTS1F"}
// SPEC: ![[FF16]] = !{i64 16, !"_ZTSM1FFivE.virtual"}
