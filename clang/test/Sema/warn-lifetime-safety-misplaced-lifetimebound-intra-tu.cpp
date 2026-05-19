// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -verify %s

struct MyObj {
  ~MyObj() {}
};

MyObj &free_param(MyObj &obj);                           // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
MyObj &free_param(MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct S {
  MyObj data;
  const MyObj &implicit_this_only();         // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
  const MyObj &param_only(const MyObj &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
  const MyObj &both(const MyObj &obj, bool); // expected-warning 2 {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
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
  T &both(T &arg, bool); // expected-warning 2 {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
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
InternalObj &anon_param(InternalObj &obj);                           // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
InternalObj &anon_param(InternalObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct AnonS {
  InternalObj data;
  InternalObj &anon_this(); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
};

InternalObj &AnonS::anon_this() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}
} // namespace

static InternalObj &static_param(InternalObj &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
static InternalObj &static_param(InternalObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct IntraSuppressedObj {
  ~IntraSuppressedObj() {}
};

IntraSuppressedObj &intra_suppressed(IntraSuppressedObj &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
IntraSuppressedObj &intra_suppressed(
    IntraSuppressedObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct View {
  friend View friend_redecl(MyObj &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}
};

// FIXME: This diagnoses an attribute inherited from another redeclaration, not one written on the definition. Once we enforce that redeclarations agree on lifetimebound, handle this with a dedicated warning and note.
View friend_redecl(MyObj &obj [[clang::lifetimebound]]); // expected-note {{'lifetimebound' attribute appears here on the definition}}

View friend_redecl(MyObj &obj) {
  return View{};
}

template <typename T>
// FIXME: Current analysis suggests adding to the primary template declaration, which is not ideal, as it will affect all specializations.
MyObj &spec_func(T &obj); // expected-warning {{'lifetimebound' attribute on this definition is not visible to callers before the definition; add it to the declaration instead}}

template <>
// FIXME: Attribute is inhetired, diagnostic's wording is not correct.
MyObj &spec_func<MyObj>(MyObj &obj [[clang::lifetimebound]]); // expected-note {{'lifetimebound' attribute appears here on the definition}}

template <>
MyObj &spec_func<MyObj>(MyObj &obj) { return obj; }
