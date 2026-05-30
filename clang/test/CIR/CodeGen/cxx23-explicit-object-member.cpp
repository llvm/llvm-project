// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Explicit-object member function with no use of `self` in the body.
// The function has two AST parameters (self, x) and no implicit `this`.
struct Functor {
  int operator()(this Functor const& self, int x) { return x + 1; }
};

int call_simple(Functor f) {
  return f(3);
}

// CIR-LABEL: cir.func{{.*}} @_ZNH7FunctorclERKS_i(%arg0: !cir.ptr<!rec_Functor>
// CIR-SAME:                                       %arg1: !s32i

// LLVM-LABEL: define {{.*}}i32 @_ZNH7FunctorclERKS_i(ptr {{.*}}%{{.+}}, i32 {{.*}}%{{.+}})

// Explicit-object member function that uses `self` to access a member.
// Confirms the explicit-object parameter is properly entered into the local
// decl map so `self.base` resolves correctly.
struct Adder {
  int base;
  int operator()(this Adder const& self, int x) { return self.base + x; }
};

int call_with_member(Adder a) {
  return a(7);
}

// CIR-LABEL: cir.func{{.*}} @_ZNH5AdderclERKS_i(%arg0: !cir.ptr<!rec_Adder>
// CIR-SAME:                                     %arg1: !s32i
// CIR:         %[[SELF_SLOT:.+]] = cir.alloca !cir.ptr<!rec_Adder>, {{.*}}["self"
// CIR:         %[[SELF_PTR:.+]] = cir.load{{.*}} %[[SELF_SLOT]] : !cir.ptr<!cir.ptr<!rec_Adder>>, !cir.ptr<!rec_Adder>
// CIR:         %[[BASE_PTR:.+]] = cir.get_member %[[SELF_PTR]][0] {name = "base"} : !cir.ptr<!rec_Adder> -> !cir.ptr<!s32i>
// CIR:         %[[BASE:.+]] = cir.load{{.*}} %[[BASE_PTR]] : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: define {{.*}}i32 @_ZNH5AdderclERKS_i(ptr {{.*}}%{{.+}}, i32 {{.*}}%{{.+}})
// LLVM:         %[[SELF_PTR:.+]] = load ptr, ptr %{{.+}}
// LLVM:         %[[BASE:.+]] = load i32, ptr %{{.+}}
// LLVM:         %[[X:.+]] = load i32, ptr %{{.+}}
// LLVM:         add nsw i32 %[[BASE]], %[[X]]

// Lambda with deducing-this explicit object parameter.
int call_lambda() {
  auto add5 = [](this auto const&, int x) { return x + 5; };
  return add5(10);
}

// CIR-LABEL: cir.func{{.*}} @_Z11call_lambdav

// LLVM-LABEL: define {{.*}}i32 @_Z11call_lambdav

// Control: ordinary (implicit-`this`) instance member function still gets
// the `this` parameter prepended.
struct WithThis {
  int v;
  int regular(int x) { return v + x; }
};

int call_regular(WithThis w) {
  return w.regular(2);
}

// CIR-LABEL: cir.func{{.*}} @_ZN8WithThis7regularEi(%arg0: !cir.ptr<!rec_WithThis>
// CIR-SAME:                                          %arg1: !s32i

// LLVM-LABEL: define {{.*}}i32 @_ZN8WithThis7regularEi(ptr {{.*}}%{{.+}}, i32 {{.*}}%{{.+}})
