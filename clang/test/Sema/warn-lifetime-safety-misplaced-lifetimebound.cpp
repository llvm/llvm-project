// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-validations -Wno-dangling -verify %s

struct MyObj {
  ~MyObj() {}
};

struct S {
  MyObj data;
  const MyObj &implicit_this_only(); // expected-warning {{'lifetimebound' attribute on the definition is not visible from this declaration; add it to the declaration instead}}
  const MyObj &param_only(const MyObj &obj); // expected-warning {{'lifetimebound' attribute on the definition is not visible from this declaration; add it to the declaration instead}}
  const MyObj &both(const MyObj &obj, bool); // expected-warning 2 {{'lifetimebound' attribute on the definition is not visible from this declaration; add it to the declaration instead}}
};

const MyObj &S::implicit_this_only() [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute written on the definition is here}}
  return data;
}

const MyObj &S::param_only(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-note {{'lifetimebound' attribute written on the definition is here}}
  return obj;
}

const MyObj &S::both(
    const MyObj &obj [[clang::lifetimebound]], bool use_obj) // expected-note {{'lifetimebound' attribute written on the definition is here}}
    [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute written on the definition is here}}
  return use_obj ? obj : data;
}

template <class T>
struct MixedSpecializations {
  T data;
  T &both(T &arg, bool); // expected-warning 2 {{'lifetimebound' attribute on the definition is not visible from this declaration; add it to the declaration instead}}
};

template <>
MyObj &MixedSpecializations<MyObj>::both(
    MyObj &arg [[clang::lifetimebound]], bool use_arg) // expected-note {{'lifetimebound' attribute written on the definition is here}}
    [[clang::lifetimebound]] { // expected-note {{'lifetimebound' attribute written on the definition is here}}
  return use_arg ? arg : data;
}
