// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety-suggestions -Wlifetime-safety -Wno-dangling -I%t -verify %t/test_source.cpp

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

struct ReturnThis {
  const ReturnThis& get() const;           // expected-warning {{implicit this in cross-TU function should be marked [[clang::lifetimebound]]}}.
};

struct ReturnThisPointer {
  const ReturnThisPointer* get() const;           // expected-warning {{implicit this in cross-TU function should be marked [[clang::lifetimebound]]}}.
};


#endif // TEST_HEADER_H

//--- test_source.cpp

#include "test_header.h"

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
