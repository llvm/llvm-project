// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-lifetimebound-violation -Wno-dangling -verify %s

#include "Inputs/lifetime-analysis.h"

struct [[gsl::Owner]] MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
};

struct [[gsl::Pointer()]] View {
  View(const MyObj &); // Borrows from MyObj
  View();
  void use() const;
};

bool cond();

View not_lb(const MyObj &obj);

View lb(const MyObj &obj [[clang::lifetimebound]]);

View return_through_unannotated_passthrough(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}
  return not_lb(obj);
}

View return_through_lifetimebound_passthrough(
    const MyObj &obj [[clang::lifetimebound]]) {
  return lb(obj);
}

View lose_lb(const MyObj &obj [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}
  return not_lb(obj);
}

View return_through_alias(const MyObj &obj [[clang::lifetimebound]]) {
  const MyObj &alias = obj;
  return alias;
}

View return_alias_through_unannotated_passthrough(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}
  const MyObj &alias = obj;
  return not_lb(alias);
}

View not_lb_view(View v);

View lb_view(View v [[clang::lifetimebound]]);


View return_through_two_lifetimebound_calls(
    const MyObj &obj [[clang::lifetimebound]]) {
  return lb_view(lb(obj));
}

View return_through_nested_broken_chain(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}
  return not_lb_view(lb_view(View(obj)));
}

View return_constructed_view_through_unannotated_forwarder(
    const MyObj &obj [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}
  return not_lb_view(View(obj));
}

View return_constructed_view_through_lifetimebound_forwarder(
    const MyObj &obj [[clang::lifetimebound]]) {
  return lb_view(View(obj));
}

View verify_each_annotated_param_independently(
    const MyObj &a [[clang::lifetimebound]],
    const MyObj &b [[clang::lifetimebound]], // expected-warning {{could not verify that the return value can be lifetime bound to 'b'}}
    const MyObj &c [[clang::lifetimebound]]) { // expected-warning {{could not verify that the return value can be lifetime bound to 'c'}}
  return cond() ? a : not_lb(b);
}

View unnamed_lifetimebound_param(
    [[clang::lifetimebound]] const MyObj &) { // expected-warning {{could not verify that the return value can be lifetime bound to an unnamed parameter}}
  return View();
}

View annotated_decl_but_not_def_not_returned(const MyObj &obj [[clang::lifetimebound]]); // expected-warning {{could not verify that the return value can be lifetime bound to 'obj'}}

View annotated_decl_but_not_def_not_returned(const MyObj &obj) {
  return not_lb(obj);
}

struct BadThisReturn {
  MyObj data;

  View get() const [[clang::lifetimebound]] { // expected-warning {{could not verify that the return value can be lifetime bound to the implicit this parameter}}
    return not_lb(data);
  }
};

struct GoodThisReturn {
  MyObj data;

  View get() const [[clang::lifetimebound]] {
    return data;
  }
};

struct RedeclaredThis {
  MyObj data;
  View get() const [[clang::lifetimebound]]; // expected-warning {{could not verify that the return value can be lifetime bound to the implicit this parameter}}
};

View RedeclaredThis::get() const {
  return not_lb(data);
}

struct ThisAndParam {
  MyObj data;

  View get(const MyObj &obj [[clang::lifetimebound]]) const [[clang::lifetimebound]] { // expected-warning {{could not verify that the return value can be lifetime bound to the implicit this parameter}}
    return lb(obj);
  }
};

struct ThisAndMixedParams {
  MyObj data;

  View get(
      const MyObj &a [[clang::lifetimebound]],
      const MyObj &b,
      const MyObj &c [[clang::lifetimebound]]) const // expected-warning {{could not verify that the return value can be lifetime bound to 'c'}}
      [[clang::lifetimebound]] {                     // expected-warning {{could not verify that the return value can be lifetime bound to the implicit this parameter}}
    return cond() ? lb(a) : not_lb(b);
  }
};

struct AssignmentCorr {
  AssignmentCorr& operator=(const AssignmentCorr& other) {
    return *this;
  }
};

struct AssignementIncorr {
  AssignementIncorr& operator=(const AssignementIncorr& other) {
    AssignementIncorr tmp;
    return tmp;
  }
};

void implicit_lifetimebound_in_nested_std_namespace() {
  (void)std::basic_string_view<char>("hello");
}
