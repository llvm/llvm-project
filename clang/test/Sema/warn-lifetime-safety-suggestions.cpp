// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -fexperimental-lifetime-safety-inference -Wexperimental-lifetime-safety-suggestions -verify %s

struct MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View();
  void use() const;
};

View return_view_directly (View a) {    // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return a;                             // expected-note {{param returned here}}
}

View conditional_return_view (
    View a,         // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
    View b,         // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
    bool c) {
  View res;
  if (c)  
    res = a;                    
  else
    res = b;          
  return res;  // expected-note 2 {{param returned here}} 
}

// FIXME: Fails to generate lifetime suggestion for reference types as these are not handled currently.
MyObj& return_reference (MyObj& a, MyObj& b, bool c) {
  if(c) {
    return a;   
  }
  return b;     
}

// FIXME: Fails to generate lifetime suggestion for reference types as these are not handled currently.
View return_view_from_reference (MyObj& p) {
  return p; 
}

int* return_pointer_directly (int* a) {    // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return a;                                // expected-note {{param returned here}} 
}

MyObj* return_pointer_object (MyObj* a) {  // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return a;                                // expected-note {{param returned here}} 
}

View only_one_paramter_annotated (View a [[clang::lifetimebound]], 
  View b,         // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  bool c) {
 if(c)
  return a;
 return b;        // expected-note {{param returned here}} 
}

View reassigned_to_another_parameter (
    View a,
    View b) {     // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  a = b;
  return a;       // expected-note {{param returned here}} 
}

struct ReturnsSelf {
  const ReturnsSelf& get() const {
    return *this;
  }
};

struct ViewProvider {
  MyObj data;
  View getView() const {
    return data;
  }
};

// FIXME: Fails to generate lifetime suggestions for the implicit 'this' parameter, as this feature is not yet implemented.
void test_get_on_temporary() {
  const ReturnsSelf& s_ref = ReturnsSelf().get();
  (void)s_ref;
}

// FIXME: Fails to generate lifetime suggestions for the implicit 'this' parameter, as this feature is not yet implemented.
void test_getView_on_temporary() {
  View sv = ViewProvider{1}.getView();
  (void)sv;
}

//===----------------------------------------------------------------------===//
// Annotation Inference Test Cases
//===----------------------------------------------------------------------===//

namespace correct_order_inference {
View return_view_by_func (View a) {    // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return return_view_directly(a);      // expected-note {{param returned here}}
}

MyObj* return_pointer_by_func (MyObj* a) {         // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return return_pointer_object(a);                 // expected-note {{param returned here}} 
}
} // correct_order_inference

namespace incorrect_order_inference_view {
View return_view_callee(View a);

// FIXME: No lifetime annotation suggestion when functions are not present in the callee-before-caller pattern
View return_view_caller(View a) {
  return return_view_callee(a);
}

View return_view_callee(View a) {     // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return a;                           // expected-note {{param returned here}}
}   
} // incorrect_order_inference_view

namespace incorrect_order_inference_object {
MyObj* return_object_callee(MyObj* a);

// FIXME: No lifetime annotation suggestion warning when functions are not present in the callee-before-caller pattern
MyObj* return_object_caller(MyObj* a) {
  return return_object_callee(a);
}

MyObj* return_object_callee(MyObj* a) {     // expected-warning {{param should be marked [[clang::lifetimebound]]}}.
  return a;                           // expected-note {{param returned here}}
}   
} // incorrect_order_inference_object

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
