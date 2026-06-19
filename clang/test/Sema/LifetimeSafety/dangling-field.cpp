// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wlifetime-safety-annotation-placement -Wno-dangling -verify %s
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
  CtorInitLifetimeBound(std::string s) : view(construct_view(s)) {} // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
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
  CtorPointerField(std::string s) : ptr(s.data()) {}  // expected-warning {{stack memory associated with parameter 's' escapes to the field 'ptr' which will dangle}}
  CtorPointerField(Dummy<1> ok, const std::string& s) : ptr(s.data()) {}
  CtorPointerField(Dummy<2> ok, std::string_view view) : ptr(view.data()) {}
};

struct MemberSetters {
  std::string_view view;  // expected-note 6 {{this field dangles}}
  const char* p;          // expected-note 6 {{this field dangles}}

  void setWithParam(std::string s) {
    view = s;     // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
    p = s.data(); // expected-warning {{stack memory associated with parameter 's' escapes to the field 'p' which will dangle}}
  }

  void setWithParamAndReturn(std::string s) {
    view = s;     // expected-warning {{stack memory associated with parameter 's' escapes to the field 'view' which will dangle}}
    p = s.data(); // expected-warning {{stack memory associated with parameter 's' escapes to the field 'p' which will dangle}}
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
    p = s.data(); // expected-warning {{stack memory associated with local variable 's' escapes to the field 'p' which will dangle}}
  }
  
  void setWithLocalButMoved() {
    std::string s;
    view = s;                 // expected-warning-re {{stack memory associated with local variable 's' may escape to the field 'view' which will dangle. {{.*}} may have been moved}}
    p = s.data();             // expected-warning-re {{stack memory associated with local variable 's' may escape to the field 'p' which will dangle. {{.*}} may have been moved}}
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
    p = local.data(); // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'p' which will dangle}}
  }

  void use_after_scope() {
    {
      std::string local;
      view = local;     // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'view' which will dangle}}
      p = local.data(); // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'p' which will dangle}}
    }
    (void)view;
    (void)p;
  }

  void use_after_scope_saved_after_reassignment() {
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
};

// FIXME: Detect escape to field of field.
struct IndirectEscape{
  struct {
    const char *p;
  } b;
  
  void foo() {
    std::string s;
    b.p = s.data();
  }
};

// FIXME: Detect escape to field of field.
struct IndirectEscape2 {
  struct {
    const char *p;
  };

  void foo() {
    std::string s;
    p = s.data();
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
    field = std::make_unique<LifetimeBoundCtor>(obj); // tu-warning {{stack memory associated with parameter 'obj' escapes to the field 'field' which will dangle}}
  }
};
} // namespace MakeUnique

namespace ThisMemberAccessSpellings {
// `this->m`, `(*this).m`, and base-casts of `this`
// (`static_cast<Base*>(this)->m`, `((Base*)this)->m`) all name the same field of
// the same object, so a borrow stored through any spelling is routed to the
// field's origin and the dangling field is caught.
struct Holder {
  std::string_view view; // expected-note 2 {{this field dangles}}
  void via_arrow() {
    std::string local;
    this->view = local;   // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'view' which will dangle}}
  }
  void via_deref() {
    std::string local;
    (*this).view = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'view' which will dangle}}
  }
};

struct Base { std::string_view q; }; // expected-note 2 {{this field dangles}}
struct Derived : Base {
  void via_static_cast() {
    std::string local;
    static_cast<Base *>(this)->q = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'q' which will dangle}}
  }
  void via_cstyle_cast() {
    std::string local;
    ((Base *)this)->q = local; // expected-warning {{stack memory associated with local variable 'local' escapes to the field 'q' which will dangle}}
  }
};

// A field use via a member call through any spelling of `this` keeps the borrow
// live (use-after-scope).
struct UseViaMethod {
  std::string_view view;
  void touch() const;
  void via_arrow() {
    {
      std::string local;
      view = local; // expected-warning {{local variable 'local' does not live long enough}}
    }               // expected-note {{destroyed here}}
    this->touch();  // expected-note {{later used here}}
  }
  void via_deref() {
    {
      std::string local;
      view = local; // expected-warning {{local variable 'local' does not live long enough}}
    }               // expected-note {{destroyed here}}
    (*this).touch(); // expected-note {{later used here}}
  }
};

// FIXME: These dangling stores are missed. this-member recognition is a
// structural match on the base spelling, so a base that only sometimes denotes
// `this` (a ternary) is not recognized; and a non-`this` object's field is not
// tracked at all, because the field-origin shortcut is keyed on the FieldDecl
// alone, which uniquely identifies an object only for `this`. Once fields carry
// their own origins/loans, this-member recognition should be loan-based (does
// the base carry the `this` loan?), which would also handle
// `(cond ? this : obj)->m` and stores into any object's field.
struct Ternary {
  std::string_view view;
  void store(Ternary *obj, bool cond) {
    std::string local;
    (cond ? this : obj)->view = local; // no-warning (FIXME: should warn for the `this` case)
  }
};
struct Plain { std::string_view view; };
void local_object_field() {
  Plain h;
  {
    std::string local;
    h.view = local; // no-warning (FIXME: should warn, dangling field of a local object)
  }
  (void)h.view;
}
} // namespace ThisMemberAccessSpellings
