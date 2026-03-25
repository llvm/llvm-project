// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -Wlifetime-safety-noescape -verify %s

#include "Inputs/lifetime-analysis.h"

struct [[gsl::Owner]] MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View();
  void use() const;
};

View return_noescape_directly(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return in; // expected-note {{returned here}}
}

View return_one_of_two(
    const MyObj& a [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    const MyObj& b [[clang::noescape]],
    bool cond) {
  if (cond)
    return a; // expected-note {{returned here}}
  return View();
}

View return_both(
    const MyObj& a [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    const MyObj& b [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    bool cond) {
  if (cond)
    return a; // expected-note {{returned here}}
  return b;   // expected-note {{returned here}}
}

View mixed_noescape_lifetimebound(
    const MyObj& a [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    const MyObj& b [[clang::lifetimebound]],
    bool cond) {
  if (cond)
    return a; // expected-note {{returned here}}
  return b;
}

View mixed_only_noescape_escapes(
    const MyObj& a [[clang::noescape]],
    const MyObj& b [[clang::lifetimebound]]) {
  (void)a;
  return b;
}

View multiple_reassign(
    const MyObj& a [[clang::noescape]],
    const MyObj& b [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    const MyObj& c [[clang::noescape]], // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    bool cond) {
  View v = a;
  if (cond)
    v = b;
  else
    v = c;
  return v; // expected-note 2 {{returned here}}
}

int* return_noescape_pointer(int* p [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return p; // expected-note {{returned here}}
}

MyObj& return_noescape_reference(MyObj& r [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return r; // expected-note {{returned here}}
}

View return_via_local(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  View v = in;
  return v; // expected-note {{returned here}}
}

void use_locally(const MyObj& in [[clang::noescape]]) {
  View v = in;
  v.use();
}

View return_without_noescape(const MyObj& in) {
  return in;
}

View return_with_lifetimebound(const MyObj& in [[clang::lifetimebound]]) {
  return in;
}

void pointer_used_locally(MyObj* p [[clang::noescape]]) {
  p->id = 42;
}

// FIXME: diagnose differently when parameter has both '[[clang::noescape]]' and '[[clang::lifetimebound]]'.
View both_noescape_and_lifetimebound(
    const MyObj& in [[clang::noescape]] [[clang::lifetimebound]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return in; // expected-note {{returned here}}
}

View identity_lifetimebound(View v [[clang::lifetimebound]]) { return v; }

View escape_through_lifetimebound_call(
    const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return identity_lifetimebound(in); // expected-note {{returned here}}
}

View no_annotation_identity(View v) { return v; }

View escape_through_unannotated_call(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return no_annotation_identity(in); // expected-note {{returned here}}
}

View global_view; // expected-note {{escapes to this global storage}}

void escape_through_global_var(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  global_view = in;
}
struct ObjWithStaticField {
  static int *static_field; // expected-note {{escapes to this static storage}}
}; 
  
void escape_to_static_data_member(int *data [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  ObjWithStaticField::static_field = data;
}



void escape_through_static_local(int *data [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  static int *static_local; // expected-note {{escapes to this static storage}}
  static_local = data;
}

thread_local int *thread_local_storage; // expected-note {{escapes to this global storage}}

void escape_through_thread_local(int *data [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  // FIXME: We might want to report whenever anything derived from a 
  // noescape parameter is assigned to a global, as this is likely 
  // undesired behavior in most cases.
  thread_local_storage = data;
}

struct ObjConsumer {
  void escape_through_member(const MyObj& in [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
    member_view = in;
  }

  View member_view; // expected-note {{escapes to this field}}
};

// FIXME: Escaping through another param is not detected.
void escape_through_param(const MyObj& in, std::vector<View> &v) {
  v.push_back(in);
}

View reassign_to_second(
    const MyObj& a [[clang::noescape]],
    const MyObj& b [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  View v = a;
  v = b;
  return v; // expected-note {{returned here}}
}

struct Container {
  MyObj data;
  const MyObj& getRef() const [[clang::lifetimebound]] { return data; }
};

View access_noescape_field(
    const Container& c [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return c.data; // expected-note {{returned here}}
}

View access_noescape_through_getter(
    Container& c [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return c.getRef(); // expected-note {{returned here}}
}

MyObj* return_ptr_from_noescape_ref(
    MyObj& r [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return &r; // expected-note {{returned here}}
}

MyObj& return_ref_from_noescape_ptr(
    MyObj* p [[clang::noescape]]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return *p; // expected-note {{returned here}}
}

int* return_spaced_brackets(int* p [ [clang::noescape] /*some comment*/ ]) { // expected-warning {{parameter is marked [[clang::noescape]] but escapes}}
  return p; // expected-note {{returned here}}
}
