// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

#include "Inputs/lifetime-analysis.h"

template<int N> struct Dummy {};
static std::string kGlobal = "GLOBAL";
void takeString(std::string&& s);

std::string_view construct_view(const std::string& str [[clang::lifetimebound]]);

struct CtorInit {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorInit(std::string s) : view(s) {} // expected-warning {{address of stack memory escapes to a field}}
};

struct CtorSet {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorSet(std::string s) { view = s; } // expected-warning {{address of stack memory escapes to a field}}
};

struct CtorInitLifetimeBound {
  std::string_view view;  // expected-note {{this field dangles}}
  CtorInitLifetimeBound(std::string s) : view(construct_view(s)) {} // expected-warning {{address of stack memory escapes to a field}}
};

struct CtorInitButMoved {
  std::string_view view;
  CtorInitButMoved(std::string s) : view(s) { takeString(std::move(s)); }
};

struct CtorInitButMovedOwned {
  std::string owned;
  std::string_view view;
  CtorInitButMovedOwned(std::string s) : view(s), owned(std::move(s)) {}
  CtorInitButMovedOwned(Dummy<1>, std::string s) : owned(std::move(s)), view(owned) {}
};

struct CtorInitMultipleViews {
  std::string_view view1; // expected-note {{this field dangles}}
  std::string_view view2; // expected-note {{this field dangles}}
  CtorInitMultipleViews(std::string s) : view1(s),   // expected-warning {{address of stack memory escapes to a field}}
                                         view2(s) {} // expected-warning {{address of stack memory escapes to a field}}
};

struct CtorInitMultipleParams {
  std::string_view view1; // expected-note {{this field dangles}}
  std::string_view view2; // expected-note {{this field dangles}}
  CtorInitMultipleParams(std::string s1, std::string s2) : view1(s1),   // expected-warning {{address of stack memory escapes to a field}}
                                                           view2(s2) {} // expected-warning {{address of stack memory escapes to a field}}
};

struct CtorRefField {
  const std::string& str;       // expected-note {{this field dangles}}
  const std::string_view& view; // expected-note {{this field dangles}}
  CtorRefField(std::string s, std::string_view v) : str(s),     // expected-warning {{address of stack memory escapes to a field}}
                                                    view(v) {}  // expected-warning {{address of stack memory escapes to a field}}
  CtorRefField(Dummy<1> ok, const std::string& s, const std::string_view& v): str(s), view(v) {}
};

struct CtorPointerField {
  const char* ptr; // expected-note {{this field dangles}}
  CtorPointerField(std::string s) : ptr(s.data()) {}  // expected-warning {{address of stack memory escapes to a field}}
  CtorPointerField(Dummy<1> ok, const std::string& s) : ptr(s.data()) {}
  CtorPointerField(Dummy<2> ok, std::string_view view) : ptr(view.data()) {}
};

struct MemberSetters {
  std::string_view view;  // expected-note 5 {{this field dangles}}
  const char* p;          // expected-note 5 {{this field dangles}}

  void setWithParam(std::string s) {
    view = s;     // expected-warning {{address of stack memory escapes to a field}}
    p = s.data(); // expected-warning {{address of stack memory escapes to a field}}
  }

  void setWithParamAndReturn(std::string s) {
    view = s;     // expected-warning {{address of stack memory escapes to a field}}
    p = s.data(); // expected-warning {{address of stack memory escapes to a field}}
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
    view = s;     // expected-warning {{address of stack memory escapes to a field}}
    p = s.data(); // expected-warning {{address of stack memory escapes to a field}}
  }
  
  void setWithLocalButMoved() {
    std::string s;
    view = s;
    p = s.data();
    takeString(std::move(s));
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
    view = local;     // expected-warning {{address of stack memory escapes to a field}}
    p = local.data(); // expected-warning {{address of stack memory escapes to a field}}
  }

  void use_after_scope() {
    {
      std::string local;
      view = local;     // expected-warning {{address of stack memory escapes to a field}}
      p = local.data(); // expected-warning {{address of stack memory escapes to a field}}
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
