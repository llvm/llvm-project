// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-suggestions -Wlifetime-safety -Wno-dangling -I%t -I%S -verify %t/test_source.cpp
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-suggestions -Wlifetime-safety -Wno-dangling -I%t -I%S -verify %t/test_source.cpp
// RUN: %clang_cc1 -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-suggestions -Wlifetime-safety -Wno-dangling -I%t -I%S -fixit %t/test_source.cpp
// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-suggestions -Wno-dangling -I%t -I%S -Werror=lifetime-safety-suggestions %t/test_source.cpp

View definition_before_header(View a);

//--- test_header.h
#ifndef TEST_HEADER_H
#define TEST_HEADER_H

struct View;

struct [[gsl::Owner]] MyObj {
  int id;
  MyObj(int i) : id(i) {} 
  MyObj() {}
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);

  View getView() const [[clang::lifetimebound]];
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View();
  void use() const;
};

View definition_before_header(View a); // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}

View return_view_directly(View a); // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}

View conditional_return_view(
  View a,        // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}
  View b,        // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}
  bool c);

int* return_pointer_directly(int* a);   // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}

MyObj* return_pointer_object(MyObj* a); // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}

inline View inline_header_return_view(View a) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return a;                                      // expected-note {{param returned here}}
}

View redeclared_in_header(View a);
inline View redeclared_in_header(View a) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return a;                                 // expected-note {{param returned here}}
}

View return_unnamed_view(View);                // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}
MyObj& return_unnamed_ref(MyObj&, bool);       // expected-warning {{parameter in cross-TU function should be marked [[clang::lifetimebound]]}}

struct ReturnThis {
  const ReturnThis& get() const;           // expected-warning {{implicit this in cross-TU function should be marked [[clang::lifetimebound]]}}.
};

struct ReturnThisPointer {
  const ReturnThisPointer* get() const;           // expected-warning {{implicit this in cross-TU function should be marked [[clang::lifetimebound]]}}.
};


#endif // TEST_HEADER_H

//--- test_source.cpp

#include "test_header.h"
#include "Inputs/lifetime-analysis.h"

View definition_before_header(View a) {
  return a;                               // expected-note {{param returned here}}
}

View return_view_directly(View a) {
  return a;                             // expected-note {{param returned here}}
}

View conditional_return_view(View a, View b, bool c) {
  View res;
  if (c)  
    res = a;                    
  else
    res = b;          
  return res;  // expected-note 2 {{param returned here}} 
}

int* return_pointer_directly(int* a) {   
  return a;                                // expected-note {{param returned here}} 
}

MyObj* return_pointer_object(MyObj* a) {
  return a;                                // expected-note {{param returned here}} 
}

View return_unnamed_view(View a) {
  return a;                               // expected-note {{param returned here}}
}

MyObj& return_unnamed_ref(MyObj& a, bool c) {
  return a;                               // expected-note {{param returned here}}
}

MyObj& return_reference(MyObj& a, // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
                        MyObj& b, // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
                        bool c) {
  if(c) {
    return a; // expected-note {{param returned here}}
  }
  return b;   // expected-note {{param returned here}}   
}

const MyObj& return_reference_const(const MyObj& a) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return a; // expected-note {{param returned here}}
}

MyObj* return_ptr_to_ref(MyObj& a) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return &a; // expected-note {{param returned here}}
}

MyObj& return_ref_to_ptr(MyObj* a) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return *a;  // expected-note {{param returned here}}
}

View return_ref_to_ptr_multiple(MyObj* a) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return *(&(*(&(*a))));  // expected-note {{param returned here}}
}

View return_view_from_reference(MyObj& p) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return p;  // expected-note {{param returned here}}
}

struct Container {  
  MyObj data;
  const MyObj& getData() [[clang::lifetimebound]] { return data; }
};

View return_struct_field(const Container& c) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return c.data; // expected-note {{param returned here}}
}
View return_struct_lifetimebound_getter(Container& c) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return c.getData().getView();  // expected-note {{param returned here}}
}

View return_view_from_reference_lifetimebound_member(MyObj& p) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return p.getView();  // expected-note {{param returned here}}
}


View return_cross_tu_func_View(View a) {      // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_view_directly(a);             // expected-note {{param returned here}} 
}

MyObj* return_cross_tu_func_obj(MyObj* a) {   // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_pointer_object(a);            // expected-note {{param returned here}} 
}

int* return_cross_tu_func_pointer(int* a) {   // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_pointer_directly(a);          // expected-note {{param returned here}} 
}

namespace {
View only_one_paramter_annotated(View a [[clang::lifetimebound]], 
  View b,         // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  bool c) {
 if(c)
  return a;
 return b;        // expected-note {{param returned here}} 
}

View reassigned_to_another_parameter(
    View a,
    View b) {     // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  a = b;
  return a;       // expected-note {{param returned here}} 
}

View intra_tu_func_redecl(View a);
View intra_tu_func_redecl(View a) {   // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                           // expected-note {{param returned here}} 
}
}

static View return_view_static(View a) {  // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                               // expected-note {{param returned here}} 
}

const ReturnThis& ReturnThis::get() const {
  return *this;                       // expected-note {{param returned here}}
}

const ReturnThisPointer* ReturnThisPointer::get() const {
  return this;                       // expected-note {{param returned here}}
}

struct ReturnsSelf {
  ReturnsSelf() {}
  ~ReturnsSelf() {}
  const ReturnsSelf& get() const { // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return *this;                  // expected-note {{param returned here}}
  }
};

struct ReturnThisAnnotated {
  const ReturnThisAnnotated& get() [[clang::lifetimebound]] { return *this; }
};

struct ViewProvider {
  ViewProvider(int d) : data(d) {}
  ~ViewProvider() {}
  MyObj data;
  View getView() const {        // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return data;                // expected-note {{param returned here}}
  }
};

View return_view_field(const ViewProvider& v) {    // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return v.data;                                   // expected-note {{param returned here}}
}

void test_get_on_temporary_pointer() {
  const ReturnsSelf* s_ref = &ReturnsSelf().get(); // expected-warning {{object whose reference is captured does not live long enough}}.
                                                   // expected-note@-1 {{destroyed here}}
  (void)s_ref;                                     // expected-note {{later used here}}
}

void test_get_on_temporary_ref() {
  const ReturnsSelf& s_ref = ReturnsSelf().get();  // expected-warning {{object whose reference is captured does not live long enough}}.
                                                   // expected-note@-1 {{destroyed here}}
  (void)s_ref;                                     // expected-note {{later used here}}
}

void test_getView_on_temporary() {
  View sv = ViewProvider{1}.getView();      // expected-warning {{object whose reference is captured does not live long enough}}.
                                            // expected-note@-1 {{destroyed here}}
  (void)sv;                                 // expected-note {{later used here}}
}

void test_get_on_temporary_copy() {
  ReturnsSelf copy = ReturnsSelf().get();                                               
  (void)copy;                                     
}

struct MemberReturn {
  MyObj data;

  MyObj& getRef() {                // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return data;                   // expected-note {{param returned here}}
  }

  MyObj& getRefExplicit() {        // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return this->data;             // expected-note {{param returned here}}
  }

  MyObj& getRefDereference() {     // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return (*this).data;           // expected-note {{param returned here}}
  }

  const MyObj* getPtr() {          // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return &data;                  // expected-note {{param returned here}}
  }

  const MyObj* getPtrExplicit() {      // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}.
    return &(this->data);              // expected-note {{param returned here}}
  }
};

//===----------------------------------------------------------------------===//
// Annotation Inference Test Cases
//===----------------------------------------------------------------------===//

namespace correct_order_inference {
View return_view_by_func(View a) {    // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_view_directly(a);      // expected-note {{param returned here}}
}

MyObj* return_pointer_by_func(MyObj* a) {         // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_pointer_object(a);                 // expected-note {{param returned here}} 
}
} // namespace correct_order_inference

namespace incorrect_order_inference_view {
View return_view_callee(View a);

View return_view_caller(View a) {     // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_view_callee(a);       // expected-note {{param returned here}}
}

View return_view_callee(View a) {     // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                           // expected-note {{param returned here}}
}   
} // namespace incorrect_order_inference_view

namespace incorrect_order_inference_object {
MyObj* return_object_callee(MyObj* a);

MyObj* return_object_caller(MyObj* a) {      // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return return_object_callee(a);            // expected-note {{param returned here}}
}

MyObj* return_object_callee(MyObj* a) {      // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                                  // expected-note {{param returned here}}
}   
} // namespace incorrect_order_inference_object

namespace simple_annotation_inference {
View inference_callee_return_identity(View a) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                                     // expected-note {{param returned here}}
}

View inference_caller_forwards_callee(View a) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return inference_callee_return_identity(a);   // expected-note {{param returned here}}
}

View inference_top_level_return_stack_view() {
  MyObj local_stack;
  return inference_caller_forwards_callee(local_stack);     // expected-warning {{address of stack memory is returned later}}
                                                            // expected-note@-1 {{returned here}}
}
} // namespace simple_annotation_inference

namespace inference_in_order_with_redecls {
View inference_callee_return_identity(View a);
View inference_callee_return_identity(View a) {   // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                                       // expected-note {{param returned here}}
}

View inference_caller_forwards_callee(View a);
View inference_caller_forwards_callee(View a) {   // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return inference_callee_return_identity(a);     // expected-note {{param returned here}}
}
  
View inference_top_level_return_stack_view() {
  MyObj local_stack;
  return inference_caller_forwards_callee(local_stack);     // expected-warning {{address of stack memory is returned later}}
                                                            // expected-note@-1 {{returned here}}
}
} // namespace inference_in_order_with_redecls

namespace inference_with_templates {
template<typename T>  
T* template_identity(T* a) {            // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return a;                             // expected-note {{param returned here}}
}

template<typename T>
T* template_caller(T* a) {              // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}.
  return template_identity(a);          // expected-note {{param returned here}}
}

MyObj* test_template_inference_with_stack() {
  MyObj local_stack;
  return template_caller(&local_stack);   // expected-warning {{address of stack memory is returned later}}
                                          // expected-note@-1 {{returned here}}                                       
}
} // namespace inference_with_templates

namespace trailing_return {
struct TrailingReturn {
  TrailingReturn() {}
  ~TrailingReturn() {}
  MyObj data;

  auto get_view() -> View {                   // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return data;                              // expected-note {{param returned here}}
  }

  auto get_view_const() const -> View {       // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return data;                              // expected-note {{param returned here}}
  }

  auto get_ref() const -> const MyObj& {      // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return data;                              // expected-note {{param returned here}}
  }
};
} // namespace trailing_return

//===----------------------------------------------------------------------===//
// Negative Test Cases
//===----------------------------------------------------------------------===//

View already_annotated(View a [[clang::lifetimebound]]) {
 return a;
}

MyObj return_obj_by_value(MyObj& p) {
  return p;
}

MyObj GlobalMyObj;
View Global = GlobalMyObj;
View Reassigned(View a) {
  a = Global;
  return a;
}

namespace lambda_captures {

struct NoSuggestionForThisCapturedByLambda {
  MyObj s;
  bool cond;
  void foo() {
    auto x = [&]() { 
      return cond > 0 ?  &this->s : &s;
    };
  }
};

void Foo(int, int*, const MyObj&, View);

auto implicit_ref_capture(int integer, int* ptr,
                          const MyObj& ref, // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
                          View view) {
  return [&]() { Foo(integer, ptr, ref, view); }; // expected-warning 3 {{address of stack memory is returned later}} \
                                                  // expected-note 3 {{returned here}} \
                                                  // expected-note {{param returned here}}
}

auto implicit_value_capture(int integer,
                            int* ptr, // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
                            const MyObj& ref,
                            View view) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return [=]() { Foo(integer, ptr, ref, view); }; // expected-note 2 {{param returned here}}
}
} // namespace lambda_captures

namespace array {

struct MemberArrayReturn {
  int arr[10];

  int* getFirst() { // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return &arr[0]; // expected-note {{param returned here}}
  }

  int* getData() { // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return arr;    // expected-note {{param returned here}}
  }
  int* getLast() {   // expected-warning {{implicit this in intra-TU function should be marked [[clang::lifetimebound]]}}
    return arr + 10; // expected-note {{param returned here}}
  }
};

} // namespace array

namespace track_origins_for_lifetimebound_record_type {

struct S {
  View view;
};

S getS(const MyObj &obj [[clang::lifetimebound]]);

S forward(const MyObj &obj) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return getS(obj);           // expected-note {{param returned here}}
}

} // namespace track_origins_for_lifetimebound_record_type

namespace capturing_constructor {
struct CaptureRefToView {
  View v; // expected-note {{escapes to this field}}
  CaptureRefToView(const MyObj& obj) : v(obj) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

CaptureRefToView test_ref_to_view() {
  MyObj obj;
  CaptureRefToView x(obj); // expected-warning {{address of stack memory is returned later}}
  return x; // expected-note {{returned here}}
}

struct CaptureRefToPtr {
  const MyObj* p; // expected-note {{escapes to this field}}
  CaptureRefToPtr(const MyObj& obj) : p(&obj) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

CaptureRefToPtr test_ref_to_ptr() {
  MyObj obj;
  CaptureRefToPtr x(obj); // expected-warning {{address of stack memory is returned later}}
  return x; // expected-note {{returned here}}
}

struct CaptureViewToView {
  View v; // expected-note {{escapes to this field}}
  CaptureViewToView(View v_param) : v(v_param) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

CaptureViewToView test_view_to_view() {
  MyObj obj;
  View v(obj); // expected-warning {{address of stack memory is returned later}}
  CaptureViewToView x(v);
  return x; // expected-note {{returned here}}
}

struct CapturePtrToPtr {
  const MyObj* p; // expected-note {{escapes to this field}}
  CapturePtrToPtr(const MyObj* p_param) : p(p_param) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

CapturePtrToPtr test_ptr_to_ptr() {
  MyObj obj;
  CapturePtrToPtr x(&obj); // expected-warning {{address of stack memory is returned later}}
  return x; // expected-note {{returned here}}
}

struct CaptureRefToRef {
  const MyObj& r; // expected-note {{escapes to this field}}
  CaptureRefToRef(const MyObj& obj) : r(obj) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

CaptureRefToRef test_ref_to_ref() {
  MyObj obj;
  CaptureRefToRef x(obj); // expected-warning {{address of stack memory is returned later}}
  return x; // expected-note {{returned here}}
}

struct BaseWithView {
  View v; // expected-note {{escapes to this field}}
};
struct CaptureRefToBaseView : BaseWithView {
  CaptureRefToBaseView(const MyObj& obj) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
    v = obj;
  }
};

CaptureRefToBaseView test_ref_to_base_view() {
  MyObj obj;
  CaptureRefToBaseView x(obj); // expected-warning {{address of stack memory is returned later}}
  return x; // expected-note {{returned here}}
}
} // namespace capturing_constructor

namespace callable_wrappers {

std::function<void()> return_lambda_capturing_param(int &x) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return [&]() { (void)x; }; // expected-note {{param returned here}}
}

void uaf_via_inferred_lifetimebound() {
  std::function<void()> f = []() {};
  {
    int local;
    f = return_lambda_capturing_param(local); // expected-warning {{object whose reference is captured does not live long enough}}
  } // expected-note {{destroyed here}}
  (void)f; // expected-note {{later used here}}
}

} // namespace callable_wrappers

namespace make_unique_suggestion {

struct LifetimeBoundCtor {
  View v;
  // FIXME: This test fails to propagate the lifetimebound in ctor if this is inferred (instead of the current explicit annotation).
  LifetimeBoundCtor(const MyObj& obj [[clang::lifetimebound]]): v(obj) {}
};

std::unique_ptr<LifetimeBoundCtor> create_target(const MyObj& obj) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return std::make_unique<LifetimeBoundCtor>(obj); // expected-note {{param returned here}}
}

void test_inference() {
  std::unique_ptr<LifetimeBoundCtor> ptr;
  {
    MyObj obj;
    ptr = create_target(obj); // expected-warning {{object whose reference is captured does not live long enough}}
  } // expected-note {{destroyed here}}
  (void)ptr; // expected-note {{later used here}}
}
} // namespace make_unique_suggestion

namespace new_allocation_suggestion {

View* MakeView(const MyObj& in) { // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
  return new View(in);            // expected-note {{param returned here}} {{destroyed here}}
}

void test_new_allocation() {
  View* v = MakeView(MyObj{}); // expected-warning {{object whose reference is captured does not live long enough}} \
                               // expected-note {{destroyed here}}
  (void)v;                     // expected-note {{later used here}}
}

struct LifetimeBoundCtor {
  View v;
  LifetimeBoundCtor();
  LifetimeBoundCtor(const MyObj& obj [[clang::lifetimebound]]) : v(obj) {}
};

struct HasCtorField {
  LifetimeBoundCtor* field;                                             // expected-note {{escapes to this field}}
  HasCtorField(const MyObj& obj) : field(new LifetimeBoundCtor(obj)) {} // expected-warning {{parameter in intra-TU function should be marked [[clang::lifetimebound]]}}
};

HasCtorField test_dangling_field_ctor() {
  MyObj obj;
  HasCtorField x(obj); // expected-warning {{address of stack memory is returned later}}
  return x;            // expected-note {{returned here}}
}

struct HasSetterField {
  LifetimeBoundCtor* field; // expected-note {{this field dangles}}
  // FIXME: Does not currently suggest `lifetime_capture_by(this)` (even without `new`)
  void set(const MyObj& obj) {
    field = new LifetimeBoundCtor(obj);
  }
  void reset() {
    MyObj obj;
    field = new LifetimeBoundCtor(obj); // expected-warning {{address of stack memory escapes to a field}}
  }
};

HasSetterField test_dangling_field_member_fn() {
  MyObj obj;
  HasSetterField x;
  x.set(obj);
  return x;
}

} // namespace new_allocation_suggestion

namespace GH193747 {

std::unique_ptr<int> create_up();
std::shared_ptr<int> create_sp() {
  return create_up();
}

struct S {
  int* field;
  S(std::unique_ptr<int>&& up) : field(up.get()) { up.release(); }
};

S foo() {
  return S(create_up());
}
} // namespace GH193747
