// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// C++23 static call operator (P1169R4) and static subscript operator
// (P2589R1).  Both produce CXXOperatorCallExpr nodes whose callee
// resolves to a static CXXMethodDecl.  Such calls must be emitted as
// ordinary static-function calls (no `this` argument) rather than
// routed through the member-call path.

struct CallFunctor {
  static int operator()(int x, int y) { return x + y; }
};

struct SubscriptFunctor {
  static int operator[](int x, int y) { return x + y; }
};

CallFunctor make_functor() { return {}; }

// Object form: `f(...)` where `f` is a named CallFunctor.

int call_static_call_operator() {
  CallFunctor f;
  return f(101, 102);
}

// CIR-LABEL: cir.func{{.*}} @_Z25call_static_call_operatorv()
// CIR:         %{{.+}} = cir.call @_ZN11CallFunctorclEii(%{{.+}}, %{{.+}})
// CIR-NOT:     cir.call @_ZN11CallFunctorclEii(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}})

// LLVM-LABEL: define {{.*}} i32 @_Z25call_static_call_operatorv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN11CallFunctorclEii(i32 {{.*}}101, i32 {{.*}}102)

// Temporary form: `CallFunctor{}(...)` — exercises the emitIgnoredExpr
// path for the object expression (ctor of the temporary).

int call_static_call_on_temporary() {
  return CallFunctor{}(201, 202);
}

// CIR-LABEL: cir.func{{.*}} @_Z29call_static_call_on_temporaryv()
// CIR:         %{{.+}} = cir.call @_ZN11CallFunctorclEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z29call_static_call_on_temporaryv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN11CallFunctorclEii(i32 {{.*}}201, i32 {{.*}}202)

// Returned-temporary form: `make_functor()(...)`.  This is the
// common shape in libcxx where a projection / comparator is built
// and immediately applied.  The object expression has a real side
// effect — emitIgnoredExpr must still call make_functor() before
// the static-operator call runs.

int call_static_call_on_returned_temp() {
  return make_functor()(301, 302);
}

// CIR-LABEL: cir.func{{.*}} @_Z33call_static_call_on_returned_tempv()
// CIR:         cir.call @_Z12make_functorv()
// CIR:         %{{.+}} = cir.call @_ZN11CallFunctorclEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z33call_static_call_on_returned_tempv()
// LLVM:         call {{.*}} @_Z12make_functorv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN11CallFunctorclEii(i32 {{.*}}301, i32 {{.*}}302)

// Member-call syntax: `f.operator()(...)`.  Goes through
// emitCXXMemberCallExpr (CXXMemberCallExpr, not CXXOperatorCallExpr),
// but should still emit a 2-arg static call — verifies that the
// member-call dispatcher also handles static call operators.

int call_static_call_member_form() {
  CallFunctor f;
  return f.operator()(801, 802);
}

// CIR-LABEL: cir.func{{.*}} @_Z28call_static_call_member_formv()
// CIR:         %{{.+}} = cir.call @_ZN11CallFunctorclEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z28call_static_call_member_formv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN11CallFunctorclEii(i32 {{.*}}801, i32 {{.*}}802)

// Qualified form: `Functor::operator()(...)`.  This is a plain
// CallExpr (no implicit object), so it falls through to emitCall
// without going through emitCallExpr's CXXOperatorCallExpr branch.

int call_static_call_qualified_form() {
  return CallFunctor::operator()(901, 902);
}

// CIR-LABEL: cir.func{{.*}} @_Z31call_static_call_qualified_formv()
// CIR:         %{{.+}} = cir.call @_ZN11CallFunctorclEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z31call_static_call_qualified_formv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN11CallFunctorclEii(i32 {{.*}}901, i32 {{.*}}902)

// Static subscript: object form.

int call_static_subscript_operator() {
  SubscriptFunctor f;
  return f[401, 402];
}

// CIR-LABEL: cir.func{{.*}} @_Z30call_static_subscript_operatorv()
// CIR:         %{{.+}} = cir.call @_ZN16SubscriptFunctorixEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z30call_static_subscript_operatorv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN16SubscriptFunctorixEii(i32 {{.*}}401, i32 {{.*}}402)

// Static subscript: temporary form.

int call_static_subscript_on_temporary() {
  return SubscriptFunctor{}[501, 502];
}

// CIR-LABEL: cir.func{{.*}} @_Z34call_static_subscript_on_temporaryv()
// CIR:         %{{.+}} = cir.call @_ZN16SubscriptFunctorixEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z34call_static_subscript_on_temporaryv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZN16SubscriptFunctorixEii(i32 {{.*}}501, i32 {{.*}}502)

// Static lambda: `[](int, int) static {...}` produces a closure type
// with a static operator(); the call site is a CXXOperatorCallExpr
// that resolves to that static operator().

auto get_static_lambda() { return [](int x, int y) static { return x + y; }; }

int call_static_lambda_operator() {
  return get_static_lambda()(601, 602);
}

// CIR-LABEL: cir.func{{.*}} @_Z27call_static_lambda_operatorv()
// CIR:         cir.call @_Z17get_static_lambdav()
// CIR:         cir.call @{{.*}}get_static_lambda{{.*}}clEii(%{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z27call_static_lambda_operatorv()
// LLVM:         call {{.*}}@_Z17get_static_lambdav()
// LLVM:         call {{.*}}get_static_lambda{{.*}}clEii{{.*}}601{{.*}}602

// Sanity: an instance call operator on the same TU still routes through
// the member-call path and gets a `this` pointer as the first argument.

struct InstanceFunctor {
  int operator()(int x, int y) const { return x - y; }
};

int call_instance_call_operator() {
  InstanceFunctor f;
  return f(701, 702);
}

// CIR-LABEL: cir.func{{.*}} @_Z27call_instance_call_operatorv()
// CIR:         %{{.+}} = cir.call @_ZNK15InstanceFunctorclEii(%{{.+}}, %{{.+}}, %{{.+}})

// LLVM-LABEL: define {{.*}} i32 @_Z27call_instance_call_operatorv()
// LLVM:         %{{.+}} = call {{.*}}i32 @_ZNK15InstanceFunctorclEii(ptr {{.*}} %{{.+}}, i32 {{.*}}701, i32 {{.*}}702)
