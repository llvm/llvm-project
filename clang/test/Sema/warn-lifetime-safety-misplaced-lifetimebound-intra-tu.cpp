// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-intra-tu-misplaced-lifetimebound -Wno-dangling -verify=intra %s

struct MyObj {
  ~MyObj() {}
};

MyObj &free_param(MyObj &obj); // intra-warning {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
MyObj &free_param(MyObj &obj [[clang::lifetimebound]]) { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct S {
  MyObj data;
  const MyObj &implicit_this_only(); // intra-warning {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
  const MyObj &param_only(const MyObj &obj); // intra-warning {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
  const MyObj &both(const MyObj &obj, bool); // intra-warning 2 {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
};

const MyObj &S::implicit_this_only() [[clang::lifetimebound]] { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return data;
}

const MyObj &S::param_only(
    const MyObj &obj [[clang::lifetimebound]]) { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

const MyObj &S::both(
    const MyObj &obj [[clang::lifetimebound]], bool use_obj) // intra-note {{'lifetimebound' attribute appears here on the definition}}
    [[clang::lifetimebound]] { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return use_obj ? obj : data;
}

template <class T>
struct MixedSpecializations {
  T data;
  T &both(T &arg, bool); // intra-warning 2 {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
};

template <>
MyObj &MixedSpecializations<MyObj>::both(
    MyObj &arg [[clang::lifetimebound]], bool use_arg) // intra-note {{'lifetimebound' attribute appears here on the definition}}
    [[clang::lifetimebound]] { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return use_arg ? arg : data;
}

struct InternalObj {
  ~InternalObj() {}
};

namespace {
InternalObj &anon_param(InternalObj &obj);
InternalObj &anon_param(InternalObj &obj [[clang::lifetimebound]]) {
  return obj;
}

struct AnonS {
  InternalObj data;
  InternalObj &anon_this();
};

InternalObj &AnonS::anon_this() [[clang::lifetimebound]] {
  return data;
}
} // namespace

static InternalObj &static_param(InternalObj &obj);
static InternalObj &static_param(InternalObj &obj [[clang::lifetimebound]]) {
  return obj;
}

struct IntraSuppressedObj {
  ~IntraSuppressedObj() {}
};

IntraSuppressedObj &intra_suppressed(IntraSuppressedObj &obj); // intra-warning {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
IntraSuppressedObj &intra_suppressed(
    IntraSuppressedObj &obj [[clang::lifetimebound]]) { // intra-note {{'lifetimebound' attribute appears here on the definition}}
  return obj;
}

struct View {
  friend View friend_redecl(MyObj &obj); // intra-warning {{'lifetimebound' attribute on an intra-TU definition is not visible to callers; add it to the declaration instead}}
};

// FIXME: Fix warning location, add a note pointing to this declaration saying "attribute inherited from this declaration"
View friend_redecl(MyObj &obj [[clang::lifetimebound]]); // intra-note {{'lifetimebound' attribute appears here on the definition}}

View friend_redecl(MyObj &obj) {
  return View{};
}
