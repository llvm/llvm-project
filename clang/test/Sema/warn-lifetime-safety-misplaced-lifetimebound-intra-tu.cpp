// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: cp %s %t.intra.cpp
// RUN: %clang_cc1 -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -fixit %t.intra.cpp
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -Werror %t.intra.cpp

struct MyObj {
  ~MyObj() {}
};

MyObj &free_param(MyObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
);

MyObj &free_param(MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct S {
  MyObj data;
  const MyObj &implicit_this_only(
  );  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:{{[0-9]+}}-[[@LINE-1]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
  
  const MyObj &param_only(const MyObj & // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
  );
  
  const MyObj &both(const MyObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      obj,  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
      bool
  );  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:{{[0-9]+}}-[[@LINE-1]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
};

const MyObj &S::implicit_this_only() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}

const MyObj &S::param_only(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

const MyObj &S::both(
    const MyObj &obj [[clang::lifetimebound]], bool use_obj) // expected-note {{'lifetimebound' attribute appears here on the definition}}
    [[clang::lifetimebound]] {                               // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return use_obj ? obj : data;
}

template <class T>
struct MixedSpecializations {
  T data;
  T &both(T &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      arg,  // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
      bool
  );  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      // CHECK-DAG: fix-it:"{{.*}}":{[[@LINE-1]]:{{[0-9]+}}-[[@LINE-1]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
};

template <>
MyObj &MixedSpecializations<MyObj>::both(
    MyObj &arg [[clang::lifetimebound]], bool use_arg) // expected-note {{'lifetimebound' attribute appears here on the definition}}
    [[clang::lifetimebound]] {                         // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return use_arg ? arg : data;
}

struct InternalObj {
  ~InternalObj() {}
};

namespace {
InternalObj &anon_param(InternalObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
);

InternalObj &anon_param(InternalObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct AnonS {
  InternalObj data;
  InternalObj &anon_this(
  );  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:{{[0-9]+}}-[[@LINE-1]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
};

InternalObj &AnonS::anon_this() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}
} // namespace

static InternalObj &static_param(InternalObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
);

static InternalObj &static_param(InternalObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct IntraSuppressedObj {
  ~IntraSuppressedObj() {}
};

IntraSuppressedObj &intra_suppressed(IntraSuppressedObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
);

IntraSuppressedObj &intra_suppressed(
    IntraSuppressedObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct View {
  friend View friend_redecl(MyObj &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
  );
};

// FIXME: This diagnoses an attribute inherited from another redeclaration, not one written on the definition. Once we enforce that redeclarations agree on lifetimebound, handle this with a dedicated warning and note.
View friend_redecl(MyObj &obj [[clang::lifetimebound]]); // expected-note {{'lifetimebound' attribute appears here on the definition}}

View friend_redecl(MyObj &obj) {
  return View{};
}

template <typename T>
// FIXME: Current analysis suggests adding to the primary template declaration, which is not ideal, as it will affect all specializations.
MyObj &spec_func(T &  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
);

template <>
// FIXME: Attribute is inhetired, diagnostic's wording is not correct.
MyObj &spec_func<MyObj>(MyObj &obj [[clang::lifetimebound]]); // expected-note {{'lifetimebound' attribute appears here on the definition}}

template <>
MyObj &spec_func<MyObj>(MyObj &obj) { return obj; }

MyObj get_default_obj();
const MyObj &default_arg_param(const MyObj&  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
    obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
    = get_default_obj());

const MyObj &default_arg_param(const MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct Base {
  virtual const MyObj& virtual_get(const MyObj& obj) const = 0;
};

struct Derived : Base {
  auto virtual_get(const MyObj&  // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
      obj  // CHECK: fix-it:"{{.*}}":{[[@LINE]]:{{[0-9]+}}-[[@LINE]]:{{[0-9]+}}}:" {{\[\[clang::lifetimebound\]\]}}"
  ) const -> const MyObj& override;
};

auto Derived::virtual_get(const MyObj& obj [[clang::lifetimebound]]) const -> const MyObj& { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}
