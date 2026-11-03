// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wlifetime-safety-annotation-placement -Wlifetime-safety-experimental-field-origin-escape -Wno-dangling -verify=expected,escape %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety -Wno-dangling -verify=expected,tu %s

#include "Inputs/lifetime-analysis.h"

template<int N> struct Dummy {};
static std::string kGlobal = "GLOBAL";
void takeString(std::string&& s);

std::string_view construct_view(const std::string& str [[clang::lifetimebound]]);

struct CtorInit {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorInit(std::string s) : view(s) {} // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
};

struct CtorSet {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorSet(std::string s) { view = s; } // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
};

// A field escape on some-but-not-all paths (or in a loop) must still be caught:
// the field's origin only spans blocks via the exit escape, so it must survive
// the join.
struct CtorSetConditional {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorSetConditional(std::string s, bool c) {
    if (c)
      view = s; // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
  }
};

struct CtorSetInLoop {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorSetInLoop(std::string s, int n) {
    for (int i = 0; i < n; ++i)
      view = s; // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
  }
};

struct CtorInitLifetimeBound {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorInitLifetimeBound(std::string s) : view(construct_view(s)) {} // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}} \
                                                                    // expected-note {{result of call to 'construct_view' aliases the storage of parameter 's'}}
};

struct CtorInitButMoved {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorInitButMoved(std::string s)
    : view(s) {  // expected-warning-re {{stack memory associated with parameter 's' may escape to the field 'view' which will dangle. {{.*}} may have been moved}}
    takeString(std::move(s)); // expected-note {{potentially moved here}}
  }
};

struct CtorInitButMovedOwned {
  std::string_view view;  // expected-note {{this field dangles}}
  std::string owned;
  CtorInitButMovedOwned(std::string s) 
    : view(s),  // expected-warning-re {{stack memory associated with parameter 's' may escape to the field 'view' which will dangle. {{.*}} may have been moved}}
    owned(std::move(s)) {}  // expected-note {{potentially moved here}}
};

struct CtorInitButMovedOwnedOrderedCorrectly {
  std::string owned;
  std::string_view view;
  CtorInitButMovedOwnedOrderedCorrectly(std::string s) 
    :owned(std::move(s)), view(owned) {} 
};

struct CtorInitMultipleViews {
  std::string_view view1; // expected-note {{this field dangles}}
  std::string_view view2; // expected-note {{this field dangles}}
  CtorInitMultipleViews(std::string s) : view1(s),   // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view1' which will dangle}}
                                         view2(s) {} // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view2' which will dangle}}
};

struct CtorInitMultipleParams {
  std::string_view view1; // expected-note {{this field dangles}}
  std::string_view view2; // expected-note {{this field dangles}}
  CtorInitMultipleParams(std::string s1, std::string s2) : view1(s1),   // expected-warning {{stack memory associated with parameter 's1' escapes to the field 'view1' which will dangle}}
                                                           view2(s2) {} // expected-warning {{stack memory associated with parameter 's2' escapes to the field 'view2' which will dangle}}
};

struct CtorRefField {
  const std::string& str;       // expected-note {{this field dangles}}
  const std::string_view& view; // expected-note {{this field dangles}}
  CtorRefField(std::string s, std::string_view v) : str(s),     // expected-warning {{stack memory associated with parameter 's' escapes to the field 'str' which will dangle}}
                                                    view(v) {}  // expected-warning {{stack memory associated with parameter 'v' escapes to the field 'view' which will dangle}}
  CtorRefField(Dummy<1> ok, const std::string& s, const std::string_view& v): str(s), view(v) {}
};

struct CtorPointerField {
  const char* ptr; // expected-note {{this field dangles}}
  CtorPointerField(std::string s) : ptr(s.data()) {}  // expected-warning {{stack memory associated with parameter 's' escapes to the field 'ptr' which will dangle}} \
                                                      // expected-note {{result of call to 'data' aliases the storage of parameter 's'}}
  CtorPointerField(Dummy<1> ok, const std::string& s) : ptr(s.data()) {}
  CtorPointerField(Dummy<2> ok, std::string_view view) : ptr(view.data()) {}
};

struct MemberSetters {
  std::string_view view;  // expected-note 6 {{this field dangles}}
  const char* p;          // expected-note 6 {{this field dangles}}

  void setWithParam(std::string s) {
    view = s;     // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
    p = s.data(); // expected-warning {{stack memory associated with parameter 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of parameter 's'}}
  }

  void setWithParamAndReturn(std::string s) {
    view = s;     // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
    p = s.data(); // expected-warning {{stack memory associated with parameter 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of parameter 's'}}
    return;
  }

  void setWithParamOk(const std::string& s) {
    view = s;
    p = s.data();
  }

  void setWithParamOkAndReturn(const std::string& s) {
    view = s;
    p = s.data();
    return;
  }

  void setWithLocal() {
    std::string s;
    view = s;     // expected-warning {{stack memory associated with local variable 's' escapes to the field 'view' which will dangle}}
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's'}}
  }
  
  void setWithLocalButMoved() {
    std::string s;
    view = s;                 // expected-warning-re {{stack memory associated with local variable 's' may escape to the field 'view' which will dangle. {{.*}} may have been moved}}
    p = s.data();             // expected-warning-re {{stack memory associated with local variable 's' may escape to the field 'p' which will dangle. {{.*}} may have been moved}} \
                              // expected-note {{result of call to 'data' aliases the storage of local variable 's'}}
    takeString(std::move(s)); // expected-note 2 {{potentially moved here}}
  }

  void setWithGlobal() {
    view = kGlobal;
    p = kGlobal.data();
  }

  void setWithLocalThenWithGlobal() {
    std::string local;
    view = local;
    p = local.data();

    view = kGlobal;
    p = kGlobal.data();
  }

  void setWithGlobalThenWithLocal() {
    view = kGlobal;
    p = kGlobal.data();

    std::string local;
    view = local;     // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'view' which will dangle}}
    p = local.data(); // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'p' which will dangle}} \
                      // expected-note {{result of call to 'data' aliases the storage of local variable 'local'}}
  }

  void escape_to_field() {
    {
      std::string local;
      view = local;     // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'view' which will dangle}}
      p = local.data(); // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'p' which will dangle}} \
                        // expected-note {{result of call to 'data' aliases the storage of local variable 'local'}}
    }
    (void)view;         // Discarding the value is not a use.
    (void)p;
  }

  void escape_to_field_saved_after_reassignment() {
    {
      std::string local;
      view = local;
      p = local.data();
    }
    (void)view;
    (void)p;

    view = kGlobal;
    p = kGlobal.data();
  }

  // Reading the dangling field is reported as a use of it.
  void use_after_scope() {
    {
      std::string local;
      view = local;     // expected-warning {{local variable 'local' does not live long enough}}
      p = local.data(); // expected-warning {{local variable 'local' does not live long enough}} \
                        // expected-note {{result of call to 'data' aliases the storage of local variable 'local' because the implicit object parameter is inferred as lifetimebound}}
    }                   // expected-note 2 {{local variable 'local' is destroyed here}}
    use(view);          // expected-note {{later used here}}
    use(p);             // expected-note {{later used here}}

    view = kGlobal;
    p = kGlobal.data();
  }
};

// FIXME: Escape to a field of a *named* subobject is not detected; only
// anonymous struct/union subobjects are traversed.
struct IndirectEscape {
  struct {
    const char *p;
  } b;

  void foo() {
    std::string s;
    b.p = s.data();
  }
};

// FIXME: Not detected either, and this shows *why* only anonymous subobjects are
// traversed: the named members `a` and `b` share the terminal field `Inner::p`,
// so a FieldDecl-keyed origin could not tell `a.p` apart from `b.p` if it
// traversed named subobjects.
struct IndirectEscapeSameType {
  struct Inner {
    const char *p;
  };
  Inner a;
  Inner b;

  void foo() {
    std::string s;
    a.p = s.data();
  }
};

// Escape to a field of an anonymous subobject is detected (members of an
// anonymous struct/union are uniquely named within the enclosing class).
struct IndirectEscape2 {
  struct {
    const char *p; // expected-note {{this field dangles}}
  };

  void foo() {
    std::string s;
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
  }
};

// A field reached through a *reference* member is NOT a subobject of `*this`
// (the reference binds to external storage), so it is not attributed to a
// `$this` field: no false diagnostic.
struct IndirectEscapeThroughReference {
  struct Inner { const char *p; };
  Inner &r;
  IndirectEscapeThroughReference(Inner &i) : r(i) {}

  void foo() {
    std::string s;
    r.p = s.data(); // no diagnostic: r.p is not a subobject of *this
  }
};

struct NestedAnonStructEscape {
  struct {
    struct {
      const char *p; // expected-note {{this field dangles}}
    };
  };

  void foo() {
    std::string s;
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
  }
};

struct MultiFieldAnonStructEscapesOnlyWrittenField {
  struct {
    const char *p; // expected-note {{this field dangles}}
    const char *q;
  };

  void foo() {
    std::string s;
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
  }
};

struct AnonUnionEscapesOnlyWrittenField {
  union {
    const char *p; // expected-note {{this field dangles}}
    const char *q;
  };

  void foo() {
    std::string s;
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
  }
};

// FIXME: Union members alias one storage, but each keeps an independent
// FieldDecl-keyed origin, so writing both members reports both.
struct AnonUnionAliasingWarnsPerMember {
  union {
    const char *p; // expected-note {{this field dangles}}
    const char *q; // expected-note {{this field dangles}}
  };

  void foo() {
    std::string s;
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
    q = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'q' which will dangle}} \
                  // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
  }
};

// FIXME: false positive: writing a non-dangling pointer through `q` overwrites
// the shared union storage, but `p`'s origin is tracked independently so `p` is
// still reported. Union aliasing is not modeled.
struct AnonUnionSafeOverwriteStillWarns {
  union {
    const char *p; // expected-note {{this field dangles}}
    const char *q;
  };

  void foo() {
    std::string s;
    p = s.data();       // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}} \
                        // expected-note {{result of call to 'data' aliases the storage of local variable 's' because the implicit object parameter is inferred as lifetimebound}}
    q = kGlobal.data(); // safe overwrite of the shared storage; `p` is still (wrongly) reported
  }
};

namespace DanglingPointerFieldInBaseClass {
struct BaseWithPointer {
  std::string_view view; // expected-note {{this field dangles}}
};

struct DerivedWithCtor : BaseWithPointer {
  DerivedWithCtor(std::string s) {
    view = s; // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
  }
};
} // namespace DanglingPointerFieldInBaseClass

namespace callable_wrappers {

struct HasCallback {
  std::function<void()> callback; // expected-note {{this field dangles}}

  void set_callback() {
    int local;
    callback = [&local]() { (void)local; }; // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'callback' which will dangle}}
  }
};

} // namespace callable_wrappers

namespace MakeUnique {
struct MyObj {};

struct LifetimeBoundCtor {
  LifetimeBoundCtor(const MyObj& obj [[clang::lifetimebound]]);
};

struct HasUniquePtrField {
  std::unique_ptr<LifetimeBoundCtor> field; // tu-note {{this field dangles}}

  void setWithParam(MyObj obj) {
    field = std::make_unique<LifetimeBoundCtor>(obj); // tu-warning {{stack memory associated with parameter 'obj' escapes to the field 'field' which will dangle}} \
                                                      // tu-note {{result of call to 'make_unique<MakeUnique::LifetimeBoundCtor, MakeUnique::MyObj &>' aliases the storage of parameter 'obj'}}
  }
};
} // namespace MakeUnique

namespace DtorNoWarn {
struct DtorSet {
  std::string_view view;
  ~DtorSet() {
    std::string s;
    view = s;
  }
};
} // namespace DtorNoWarn

namespace LambdaCaptureReset {
struct MyObj {};
struct HasField {
  MyObj* ptr; // expected-note 3 {{this field dangles}}

  void this_capture() {
    MyObj local;
    ptr = &local; // expected-warning-re {{stack memory associated with local variable 'local' may escape to the field 'ptr' which will dangle. {{.*}} captured by a lambda}}
    auto cleanup = [this]() {
      ptr = nullptr;
    };
  }

  void capture_by_ref() {
    MyObj local;
    ptr = &local; // expected-warning-re {{stack memory associated with local variable 'local' may escape to the field 'ptr' which will dangle. {{.*}} captured by a lambda}}
    auto cleanup = [&]() {
      ptr = nullptr;
    };
  }

  void foo_init_capture() {
    MyObj local;
    ptr = &local; // expected-warning-re {{stack memory associated with local variable 'local' may escape to the field 'ptr' which will dangle. {{.*}} captured by a lambda}}
    auto cleanup = [&p = ptr]() {
      p = nullptr;
    };
  }
};
} // namespace LambdaCaptureReset

// Methods instantiated eagerly are analyzed while their class is being defined.
namespace GH236616 {
template <class T> struct NoFields {
  __attribute__((used)) constexpr int get() const { return 0; }
};
NoFields<int> no_fields;

// Taking the address of a method with a deduced return type instantiates it.
template <class T> struct DeducedReturn {
  constexpr auto get() const { return 0; }
  using R = decltype(&DeducedReturn::get);
};
DeducedReturn<int> deduced_return;

template <class T> struct WithField {
  const int *p;                                            // escape-note {{if field 'p' is owned by the class}}
  __attribute__((used)) constexpr const int *get() const { // escape-warning {{depends on the object referenced by field 'p'}}
    return p;                                              // escape-note {{returned here}}
  }
};
WithField<int> with_field; // escape-note {{in instantiation of member function}} \
                           // escape-note {{in instantiation of template class}}
} // namespace GH236616
