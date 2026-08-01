// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

int Bar(int);
int Baz(int);

int Func1(int x) {
  if (x)
    [[clang::musttail]] return Bar(x);
  else
    [[clang::musttail]] return Baz(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z5Func1i
// CIR:         %[[R1:.+]] = cir.call @_Z3Bari(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R1]] : !s32i
// CIR:         %[[R2:.+]] = cir.call @_Z3Bazi(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R2]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z5Func1i
// LLVM:         %[[C1:.+]] = musttail call{{.*}} i32 @_Z3Bari(i32{{.*}})
// LLVM-NEXT:    ret i32 %[[C1]]
// LLVM:         %[[C2:.+]] = musttail call{{.*}} i32 @_Z3Bazi(i32{{.*}})
// LLVM-NEXT:    ret i32 %[[C2]]

int Nested(int x) {
  [[clang::musttail]] return Bar(Bar(x));
}

// CIR-LABEL: cir.func{{.*}} @_Z6Nestedi
// CIR:         %[[INNER:.+]] = cir.call @_Z3Bari(%{{.+}}) : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    %[[OUTER:.+]] = cir.call @_Z3Bari(%[[INNER]]) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[OUTER]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z6Nestedi
// LLVM:         %[[INNER:.+]] = call{{.*}} i32 @_Z3Bari
// LLVM:         %[[OUTER:.+]] = musttail call{{.*}} i32 @_Z3Bari(i32{{.*}} %[[INNER]])
// LLVM-NEXT:    ret i32 %[[OUTER]]

int Scoped(int x) {
  { [[clang::musttail]] return Bar(x); }
}

// CIR-LABEL: cir.func{{.*}} @_Z6Scopedi
// CIR:         %[[R:.+]] = cir.call @_Z3Bari(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z6Scopedi
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_Z3Bari
// LLVM-NEXT:    ret i32 %[[R]]

void ReturnsVoid();
void VoidTail() {
  [[clang::musttail]] return ReturnsVoid();
}

// CIR-LABEL: cir.func{{.*}} @_Z8VoidTailv
// CIR:         cir.call @_Z11ReturnsVoidv() musttail : () -> ()
// CIR-NEXT:    cir.return

// LLVM-LABEL: define{{.*}} void @_Z8VoidTailv
// LLVM:         musttail call void @_Z11ReturnsVoidv()
// LLVM-NEXT:    ret void

void IndirectLocal(int x) {
  void (*p)(int) = nullptr;
  [[clang::musttail]] return p(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z13IndirectLocali
// CIR:         cir.call %{{.+}}(%{{.+}}) musttail : (!cir.ptr<!cir.func<(!s32i)>>, !s32i{{.*}}) -> ()
// CIR-NEXT:    cir.return

// LLVM-LABEL: define{{.*}} void @_Z13IndirectLocali
// LLVM:         musttail call void %{{.+}}(i32{{.*}})
// LLVM-NEXT:    ret void

struct Data {
  int (*fptr)(Data *);
};
int IndirectField(Data *data) {
  [[clang::musttail]] return data->fptr(data);
}

// CIR-LABEL: cir.func{{.*}} @_Z13IndirectFieldP4Data
// CIR:         %[[R:.+]] = cir.call %{{.+}}(%{{.+}}) musttail : (!cir.ptr<{{.*}}>, !cir.ptr<!rec_Data>{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z13IndirectFieldP4Data
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 %{{.+}}(ptr{{.*}})
// LLVM-NEXT:    ret i32 %[[R]]

struct Foo {
  int MemberFunction(int x);
  static int StaticMethod(int x);
  int TailFrom(int x);
  int TailFrom2(int x);
};

int Foo::TailFrom(int x) {
  [[clang::musttail]] return MemberFunction(x);
}

// CIR-LABEL: cir.func{{.*}} @_ZN3Foo8TailFromEi
// CIR:         %[[R:.+]] = cir.call @_ZN3Foo14MemberFunctionEi(%{{.+}}, %{{.+}}) musttail : (!cir.ptr<!rec_Foo>{{.*}}, !s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_ZN3Foo8TailFromEi
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_ZN3Foo14MemberFunctionEi(ptr{{.*}}, i32{{.*}})
// LLVM-NEXT:    ret i32 %[[R]]

int StaticViaClass(int x) {
  [[clang::musttail]] return Foo::StaticMethod(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z14StaticViaClassi
// CIR:         %[[R:.+]] = cir.call @_ZN3Foo12StaticMethodEi(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z14StaticViaClassi
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_ZN3Foo12StaticMethodEi(i32{{.*}})
// LLVM-NEXT:    ret i32 %[[R]]

int StaticViaObject(int x) {
  Foo foo;
  [[clang::musttail]] return foo.StaticMethod(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z15StaticViaObjecti
// CIR:         %[[R:.+]] = cir.call @_ZN3Foo12StaticMethodEi(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z15StaticViaObjecti
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_ZN3Foo12StaticMethodEi(i32{{.*}})
// LLVM-NEXT:    ret i32 %[[R]]

template <class T> int TplBody(int x) {
  T t;
  [[clang::musttail]] return Bar(x);
}
int InstBody(int x) { return TplBody<int>(x); }

// CIR-LABEL: cir.func{{.*}} @_Z7TplBodyIiEii
// CIR:         %[[R:.+]] = cir.call @_Z3Bari(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z7TplBodyIiEii
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_Z3Bari
// LLVM-NEXT:    ret i32 %[[R]]

template <class T> T TplQualified(T x) {
  [[clang::musttail]] return ::Bar(x);
}
int InstQualified(int x) { return TplQualified<int>(x); }

// CIR-LABEL: cir.func{{.*}} @_Z12TplQualifiedIiET_S0_
// CIR:         %[[R:.+]] = cir.call @_Z3Bari(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z12TplQualifiedIiET_S0_
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_Z3Bari
// LLVM-NEXT:    ret i32 %[[R]]

struct TrivialDtor {};
int ReturnsInt(int x);
int TrivialLocal(int x) {
  TrivialDtor foo;
  [[clang::musttail]] return ReturnsInt(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z12TrivialLocali
// CIR:         %[[R:.+]] = cir.call @_Z10ReturnsInti(%{{.+}}) musttail : (!s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z12TrivialLocali
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 @_Z10ReturnsInti
// LLVM-NEXT:    ret i32 %[[R]]

// FIXME(cir): This requires a cleanup scope of some sort, but is NYI.
//   int VlaScope(int x) {
//     int vla[x];
//     [[clang::musttail]] return Bar(x);
//   }

int (Foo::*pmf)(int);
int Foo::TailFrom2(int x) {
  [[clang::musttail]] return (this->*pmf)(x);
}

// CIR-LABEL: cir.func{{.*}} @_ZN3Foo9TailFrom2Ei
// CIR:         %[[R:.+]] = cir.call %{{.+}}(%{{.+}}, %{{.+}}) musttail : (!cir.ptr<{{.*}}>, !cir.ptr<!void>{{.*}}, !s32i{{.*}}) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_ZN3Foo9TailFrom2Ei
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 %{{.+}}(ptr{{.*}}, i32{{.*}})
// LLVM-NEXT:    ret i32 %[[R]]

struct V {
  virtual void t();
  virtual void f();
};
void V::f() {
  [[clang::musttail]] return t();
}

// CIR-LABEL: cir.func{{.*}} @_ZN1V1fEv
// CIR:         cir.call %{{.+}}(%{{.+}}) musttail : (!cir.ptr<{{.*}}>, !cir.ptr<!rec_V>{{.*}}) -> ()
// CIR-NEXT:    cir.return

// LLVM-LABEL: define{{.*}} void @_ZN1V1fEv
// LLVM:         musttail call void %{{.+}}(ptr{{.*}})
// LLVM-NEXT:    ret void

// FIXME(cir): Another one that requires cleanups.
//   struct NT { NT(const NT &); char d[32]; };
//   NT RCBV();
//   NT SretTail() {
//     [[clang::musttail]] return RCBV();
//   }

int Lam() {
  auto l = []() { return 12; };
  [[clang::musttail]] return (+l)();
}

// CIR-LABEL: cir.func{{.*}} @_Z3Lamv
// CIR:         %{{.+}} = cir.call @_ZZ3LamvENK3$_0cvPFivEEv(%{{.+}}){{.*}} : (!cir.ptr<{{.*}}>{{.*}}) -> (!cir.ptr<!cir.func<() -> !s32i>>{{.*}})
// CIR:         %[[R:.+]] = cir.call %{{.+}}() musttail : (!cir.ptr<!cir.func<() -> !s32i>>) -> (!s32i{{.*}})
// CIR-NEXT:    cir.return %[[R]] : !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z3Lamv
// LLVM:         %[[R:.+]] = musttail call{{.*}} i32 %{{.+}}()
// LLVM-NEXT:    ret i32 %[[R]]

