// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify=expected,function %s
// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety -Wno-dangling -verify=expected,tu %s

#include "Inputs/lifetime-analysis.h"

struct View;

struct [[gsl::Owner]] MyObj {
  int id;
  MyObj();
  MyObj(int);
  MyObj(const MyObj&);
  MyObj(MyObj&&);
  MyObj& operator=(MyObj&&);
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);
  
  View getView() const [[clang::lifetimebound]];
  const int* getData() const [[clang::lifetimebound]];
};

struct [[gsl::Owner]] MyTrivialObj {
  int id;
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View(const MyTrivialObj &); // Borrows from MyTrivialObj
  View();
  void use() const;

  const MyObj* data() const;
  const MyObj& operator*() const;
  const MyObj* operator->() const;
};

class TriviallyDestructedClass {
  View a, b;
};

MyObj non_trivially_destructed_temporary();
MyTrivialObj trivially_destructed_temporary();
View construct_view(const MyObj &obj [[clang::lifetimebound]]) {
  return View(obj);
}
void use(View);

//===----------------------------------------------------------------------===//
// Basic Use-After-Free
//===----------------------------------------------------------------------===//

void simple_case() {
  MyObj* p;
  {
    MyObj s;
    p = &s;     // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
}

void simple_case_gsl() {
  View v;
  {
    MyObj s;
    v = s;      // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note {{later used here}}
}

void no_use_no_error() {
  MyObj* p;
  {
    MyObj s;
    p = &s;
  }
  // 'p' is dangling here, but since it is never used, no warning is issued.
}

void no_use_no_error_gsl() {
  View v;
  {
    MyObj s;
    v = s;
  }
  // 'v' is dangling here, but since it is never used, no warning is issued.
}

void pointer_chain() {
  MyObj* p;
  MyObj* q;
  {
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
    q = p;
  }             // expected-note {{destroyed here}}
  (void)*q;     // expected-note {{later used here}}
}

void propagation_gsl() {
  View v1, v2;
  {
    MyObj s;
    v1 = s;     // expected-warning {{object whose reference is captured does not live long enough}}
    v2 = v1;
  }             // expected-note {{destroyed here}}
  v2.use();     // expected-note {{later used here}}
}

void multiple_uses_one_warning() {
  MyObj* p;
  {
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
  // No second warning for the same loan.
  p->id = 1;
  MyObj* q = p;
  (void)*q;
}

void multiple_pointers() {
  MyObj *p, *q, *r;
  {
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
    q = &s;     // expected-warning {{does not live long enough}}
    r = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note 3 {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
  (void)*q;     // expected-note {{later used here}}
  (void)*r;     // expected-note {{later used here}}
}

void single_pointer_multiple_loans(bool cond) {
  MyObj *p;
  if (cond){
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  else {
    MyObj t;
    p = &t;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note 2  {{later used here}}
}

void single_pointer_multiple_loans_gsl(bool cond) {
  View v;
  if (cond){
    MyObj s;
    v = s;      // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  else {
    MyObj t;
    v = t;      // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note 2 {{later used here}}
}

void if_branch(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  if (cond) {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
}

void if_branch_potential(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  if (cond) {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  if (!cond)
    (void)*p;   // expected-note {{later used here}}
  else
    p = &safe;
}

void if_branch_gsl(bool cond) {
  MyObj safe;
  View v = safe;
  if (cond) {
    MyObj temp;
    v = temp;   // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note {{later used here}}
}

void potential_together(bool cond) {
  MyObj safe;
  MyObj* p_maybe = &safe;
  MyObj* p_definite = nullptr;

  {
    MyObj s;
    if (cond)
      p_definite = &s;  // expected-warning {{does not live long enough}}
    if (cond)
      p_maybe = &s;     // expected-warning {{does not live long enough}}         
  }                     // expected-note 2 {{destroyed here}}
  (void)*p_definite;    // expected-note {{later used here}}
  if (!cond)
    (void)*p_maybe;     // expected-note {{later used here}}
}

void overrides_potential(bool cond) {
  MyObj safe;
  MyObj* p;
  MyObj* q;
  {
    MyObj s;
    q = &s;       // expected-warning {{does not live long enough}}
    p = q;
  }               // expected-note {{destroyed here}}

  if (cond) {
    // 'q' is conditionally "rescued". 'p' is not.
    q = &safe;
  }

  // The use of 'p' dominates expiry of 's' error because it was never rescued.
  (void)*q;
  (void)*p;       // expected-note {{later used here}}
  (void)*q;
}

void due_to_conditional_killing(bool cond) {
  MyObj safe;
  MyObj* q;
  {
    MyObj s;
    q = &s;       // expected-warning {{does not live long enough}}
  }               // expected-note {{destroyed here}}
  if (cond) {
    // 'q' is conditionally "rescued". 'p' is not.
    q = &safe;
  }
  (void)*q;       // expected-note {{later used here}}
}

void for_loop_use_after_loop_body(MyObj safe) {
  MyObj* p = &safe;
  for (int i = 0; i < 1; ++i) {
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
}

void safe_for_loop_gsl() {
  MyObj safe;
  View v = safe;
  for (int i = 0; i < 1; ++i) {
    MyObj s;
    v = s;
    v.use();
  }
}

void for_loop_gsl() {
  MyObj safe;
  View v = safe;
  for (int i = 0; i < 1; ++i) {
    MyObj s;
    v = s;      // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note {{later used here}}
}

void for_loop_use_before_loop_body(MyObj safe) {
  MyObj* p = &safe;
  // Prefer the earlier use for diagnsotics.
  for (int i = 0; i < 1; ++i) {
    (void)*p;   // expected-note {{later used here}}
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;
}

void loop_with_break(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  for (int i = 0; i < 10; ++i) {
    if (cond) {
      MyObj temp;
      p = &temp; // expected-warning {{does not live long enough}}
      break;     // expected-note {{destroyed here}}
    }           
  } 
  (void)*p;     // expected-note {{later used here}}
}

void loop_with_break_gsl(bool cond) {
  MyObj safe;
  View v = safe;
  for (int i = 0; i < 10; ++i) {
    if (cond) {
      MyObj temp;
      v = temp;   // expected-warning {{object whose reference is captured does not live long enough}}
      break;      // expected-note {{destroyed here}}
    }
  }
  v.use();      // expected-note {{later used here}}
}

void multiple_expiry_of_same_loan(bool cond) {
  // Choose the last expiry location for the loan (e.g., through scope-ends and break statements).
  MyObj safe;
  MyObj* p = &safe;
  for (int i = 0; i < 10; ++i) {
    MyObj unsafe;
    if (cond) {
      p = &unsafe; // expected-warning {{does not live long enough}}
      break;       // expected-note {{destroyed here}} 
    }
  }
  (void)*p;       // expected-note {{later used here}}

  p = &safe;
  for (int i = 0; i < 10; ++i) {
    MyObj unsafe;
    if (cond) {
      p = &unsafe;    // expected-warning {{does not live long enough}}
      if (cond)
        break;        // expected-note {{destroyed here}}
    }
  }
  (void)*p;           // expected-note {{later used here}}

  p = &safe;
  for (int i = 0; i < 10; ++i) {
    if (cond) {
      MyObj unsafe2;
      p = &unsafe2;   // expected-warning {{does not live long enough}}
      break;          // expected-note {{destroyed here}}
    }
  }
  (void)*p;           // expected-note {{later used here}}

  p = &safe;
  for (int i = 0; i < 10; ++i) {
    MyObj unsafe;
    if (cond)
      p = &unsafe;    // expected-warning {{does not live long enough}}
    if (cond)
      break;          // expected-note {{destroyed here}}
  }
  (void)*p;           // expected-note {{later used here}}
}

void switch_potential(int mode) {
  MyObj safe;
  MyObj* p = &safe;
  switch (mode) {
  case 1: {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  case 2: {
    p = &safe;  // This path is okay.
    break;
  }
  }
  if (mode == 2)
    (void)*p;     // expected-note {{later used here}}
}

void switch_uaf(int mode) {
  MyObj safe;
  MyObj* p = &safe;
  // A use domintates all the loan expires --> all definite error.
  switch (mode) {
  case 1: {
    MyObj temp1;
    p = &temp1; // expected-warning {{does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  case 2: {
    MyObj temp2;
    p = &temp2; // expected-warning {{does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  default: {
    MyObj temp2;
    p = &temp2; // expected-warning {{does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  }
  (void)*p;     // expected-note 3 {{later used here}}
}

void switch_gsl(int mode) {
  View v;
  switch (mode) {
  case 1: {
    MyObj temp1;
    v = temp1;  // expected-warning {{object whose reference is captured does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  case 2: {
    MyObj temp2;
    v = temp2;  // expected-warning {{object whose reference is captured does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  default: {
    MyObj temp3;
    v = temp3;  // expected-warning {{object whose reference is captured does not live long enough}}
    break;      // expected-note {{destroyed here}}
  }
  }
  v.use();      // expected-note 3 {{later used here}}
}

void loan_from_previous_iteration(MyObj safe, bool condition) {
  MyObj* p = &safe;
  MyObj* q = &safe;

  while (condition) {
    MyObj x;
    p = &x;     // expected-warning {{does not live long enough}}

    if (condition)
      q = p;
    (void)*p;
    (void)*q;   // expected-note {{later used here}}
  }             // expected-note {{destroyed here}}
}

void trivial_int_uaf() {
  int * a;
  {
      int b = 1;
      a = &b;  // expected-warning {{object whose reference is captured does not live long enough}}
  }            // expected-note {{destroyed here}}
  (void)*a;    // expected-note {{later used here}}
}

void trivial_class_uaf() {
  TriviallyDestructedClass* ptr;
  {
      TriviallyDestructedClass s;
      ptr = &s; // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)ptr;    // expected-note {{later used here}}
}

void small_scope_reference_var_no_error() {
  MyObj safe;
  View view;
  {
    const MyObj& ref = safe;
    view = ref;
  }
  view.use();
}

//===----------------------------------------------------------------------===//
// Basic Use-After-Return (Return-Stack-Address)
//===----------------------------------------------------------------------===//

MyObj* simple_return_stack_address() {
  MyObj s;      
  MyObj* p = &s; // expected-warning {{address of stack memory is returned later}}
  return p;      // expected-note {{returned here}}
}

MyObj* direct_return() {
  MyObj s;      
  return &s;     // expected-warning {{address of stack memory is returned later}}
                 // expected-note@-1 {{returned here}}
}

const MyObj& return_reference_to_param_no_error(const MyObj& in) {
  return in;
}

const MyObj& return_reference_to_param_via_ref_no_error(const MyObj& in) {
  const MyObj& ref = in;
  return ref;
}

const MyObj* getPointer();
const MyObj& return_reference_to_param_via_pointer_no_error() {
  const MyObj& ref = *getPointer();
  return ref;
}

const MyObj* conditional_assign_unconditional_return(const MyObj& safe, bool c) {
  MyObj s; 
  const MyObj* p = &safe;
  if (c) {
    p = &s;       // expected-warning {{address of stack memory is returned later}}
  }     
  return p;      // expected-note {{returned here}}
}

View conditional_assign_both_branches(const MyObj& safe, bool c) {
  MyObj s;
  View p;
  if (c) {
    p = s;      // expected-warning {{address of stack memory is returned later}}
  } 
  else {
    p = safe;
  }
  return p;     // expected-note {{returned here}}

}

View reassign_safe_to_local(const MyObj& safe) {
  MyObj local;
  View p = safe;
  p = local;    // expected-warning {{address of stack memory is returned later}}
  return p;     // expected-note {{returned here}}
}

View pointer_chain_to_local() {
  MyObj local;
  View p1 = local;     // expected-warning {{address of stack memory is returned later}}
  View p2 = p1; 
  return p2;          // expected-note {{returned here}}
}

View multiple_assign_multiple_return(const MyObj& safe, bool c1, bool c2) {
  MyObj local1;
  MyObj local2;
  View p;
  if (c1) {
    p = local1;       // expected-warning {{address of stack memory is returned later}}
    return p;         // expected-note {{returned here}}
  }
  else if (c2) {
    p = local2;       // expected-warning {{address of stack memory is returned later}}
    return p;         // expected-note {{returned here}}
  }
  p = safe;
  return p;
}

View multiple_assign_single_return(const MyObj& safe, bool c1, bool c2) {
  MyObj local1;
  MyObj local2;
  View p;
  if (c1) {
    p = local1;      // expected-warning {{address of stack memory is returned later}}
  }
  else if (c2) {
    p = local2;      // expected-warning {{address of stack memory is returned later}}
  }
  else {
    p = safe;
  }
  return p;         // expected-note 2 {{returned here}}
}

View direct_return_of_local() {
  MyObj stack;      
  return stack;     // expected-warning {{address of stack memory is returned later}}
                    // expected-note@-1 {{returned here}}
}

MyObj& reference_return_of_local() {
  MyObj stack;      
  return stack;     // expected-warning {{address of stack memory is returned later}}
                    // expected-note@-1 {{returned here}}
}

int* trivial_int_uar() {
  int *a;
  int b = 1;
  a = &b;          // expected-warning {{address of stack memory is returned later}}
  return a;        // expected-note {{returned here}}
}

TriviallyDestructedClass* trivial_class_uar () {
  TriviallyDestructedClass *ptr;
  TriviallyDestructedClass s;
  ptr = &s;       // expected-warning {{address of stack memory is returned later}}
  return ptr;     // expected-note {{returned here}}
}

const int& return_parameter(int a) { 
  return a; // expected-warning {{address of stack memory is returned later}}
            // expected-note@-1 {{returned here}}
}

int* return_pointer_to_parameter(int a) {
    return &a;  // expected-warning {{address of stack memory is returned later}}
                // expected-note@-1 {{returned here}}
}

const int& return_reference_to_parameter(int a) {
    const int &b = a;   // expected-warning {{address of stack memory is returned later}}
    return b;           // expected-note {{returned here}}
}
int return_reference_to_parameter_no_error(int a) {
    const int &b = a;
    return b;
}

MyObj*& return_ref_to_local_ptr_pointing_to_local() {
  MyObj local;
  MyObj* p = &local; // expected-warning {{address of stack memory is returned later}}
  return p;          // expected-note {{returned here}} \
                     // expected-warning {{address of stack memory is returned later}} \
                     // expected-note {{returned here}}
}

const int& reference_via_conditional(int a, int b, bool cond) {
    const int &c = (cond ? ((a)) : (b));  // expected-warning 2 {{address of stack memory is returned later}}
    return c;                             // expected-note 2 {{returned here}}
}
const int* return_pointer_to_parameter_via_reference(int a, int b, bool cond) {
    const int &c = cond ? a : b;  // expected-warning 2 {{address of stack memory is returned later}}
    const int* d = &c;
    return d;                     // expected-note 2 {{returned here}}
}

const int& return_pointer_to_parameter_via_reference_1(int a) {
    const int* d = &a; // expected-warning {{address of stack memory is returned later}}
    return *d;    // expected-note {{returned here}}
}

const int& get_ref_to_local() {
    int a = 42;
    return a;         // expected-warning {{address of stack memory is returned later}}
                      // expected-note@-1 {{returned here}}
}

void test_view_pointer() {
  View* vp;
  {
    View v;
    vp = &v;     // expected-warning {{object whose reference is captured does not live long enough}}
  }              // expected-note {{destroyed here}}
  vp->use();     // expected-note {{later used here}}
}

void test_view_double_pointer() {
  View** vpp;
  {
    View* vp = nullptr;
    vpp = &vp;   // expected-warning {{object whose reference is captured does not live long enough}}
  }              // expected-note {{destroyed here}}
  (**vpp).use(); // expected-note {{later used here}}
}

struct PtrHolder {
  int* ptr;
  int* const& getRef() const [[clang::lifetimebound]] { return ptr; }
};

int* const& test_ref_to_ptr() {
  PtrHolder a;
  int *const &ref = a.getRef();  // expected-warning {{address of stack memory is returned later}}
  return ref;  // expected-note {{returned here}}
}
int* const test_ref_to_ptr_no_error() {
  PtrHolder a;
  int *const &ref = a.getRef();
  return ref;
}

int** return_inner_ptr_addr(int*** ppp [[clang::lifetimebound]]);
void test_lifetimebound_multi_level() {
  int** result;
  {
    int* p = nullptr;
    int** pp = &p;  
    int*** ppp = &pp; // expected-warning {{object whose reference is captured does not live long enough}}
    result = return_inner_ptr_addr(ppp);
  }                   // expected-note {{destroyed here}}
  (void)**result;     // expected-note {{used here}}
}

// FIXME: Assignment does not track the dereference of a pointer.
void test_assign_through_double_ptr() {
  int a = 1, b = 2;
  int* p = &a;
  int** pp = &p;
  {
    int c = 3;
    *pp = &c;
  }
  (void)**pp;
}

int** test_ternary_double_ptr(bool cond) {
  int a = 1, b = 2;
  int* pa = &a;  // expected-warning {{address of stack memory is returned later}}
  int* pb = &b;  // expected-warning {{address of stack memory is returned later}}
  int** result = cond ? &pa : &pb;  // expected-warning 2 {{address of stack memory is returned later}}
  return result; // expected-note 4 {{returned here}}
}
//===----------------------------------------------------------------------===//
// Use-After-Scope & Use-After-Return (Return-Stack-Address) Combined
// These are cases where the diagnostic kind is determined by location
//===----------------------------------------------------------------------===//

MyObj* uaf_before_uar() {
  MyObj* p;
  {
    MyObj local_obj; 
    p = &local_obj;  // expected-warning {{object whose reference is captured does not live long enough}}
  }                  // expected-note {{destroyed here}}
  return p;          // expected-note {{later used here}}
}

View uar_before_uaf(const MyObj& safe, bool c) {
  View p;
  {
    MyObj local_obj; 
    p = local_obj;  // expected-warning {{ddress of stack memory is returned later}}
    if (c) {
      return p;     // expected-note {{returned here}}
    }
  }
  p.use();
  p = safe;
  return p;
}

//===----------------------------------------------------------------------===//
// No-Error Cases
//===----------------------------------------------------------------------===//
void no_error_if_dangle_then_rescue() {
  MyObj safe;
  MyObj* p;
  {
    MyObj temp;
    p = &temp;  // p is temporarily dangling.
  }
  p = &safe;    // p is "rescued" before use.
  (void)*p;     // This is safe.
}

void no_error_if_dangle_then_rescue_gsl() {
  MyObj safe;
  View v;
  {
    MyObj temp;
    v = temp;  // 'v' is temporarily dangling.
  }
  v = safe;    // 'v' is "rescued" before use by reassigning to a valid object.
  v.use();     // This is safe.
}

void no_error_if_dangle_then_rescue_via_ref() {
  MyObj safe;
  MyObj* p;
  MyObj*& ref = p;
  {
    MyObj temp;
    ref = &temp;  // p temporarily points to temp via ref.
  }
  ref = &safe;    // p is "rescued" via ref before use.
  (void)*ref;     // This is safe.
}

void no_error_loan_from_current_iteration(bool cond) {
  // See https://github.com/llvm/llvm-project/issues/156959.
  MyObj b;
  while (cond) {
    MyObj a;
    View p = b;
    if (cond) {
      p = a;
    }
    (void)p;
  }
}

View safe_return(const MyObj& safe) {
  MyObj local;
  View p = local;
  p = safe;     // p has been reassigned
  return p;     // This is safe
}

//===----------------------------------------------------------------------===//
// Lifetimebound Attribute Tests
//===----------------------------------------------------------------------===//

View Identity(View v [[clang::lifetimebound]]);
const MyObj& IdentityRef(const MyObj& obj [[clang::lifetimebound]]);
MyObj* Identity(MyObj* v [[clang::lifetimebound]]);
View Choose(bool cond, View a [[clang::lifetimebound]], View b [[clang::lifetimebound]]);
MyObj* GetPointer(const MyObj& obj [[clang::lifetimebound]]);

void lifetimebound_simple_function() {
  View v;
  {
    MyObj obj;
    v = Identity(obj); // expected-warning {{object whose reference is captured does not live long enough}}
  }                    // expected-note {{destroyed here}}
  v.use();             // expected-note {{later used here}}
}

void lifetimebound_multiple_args_definite() {
  View v;
  {
    MyObj obj1, obj2;
    v = Choose(true,
               obj1,  // expected-warning {{object whose reference is captured does not live long enough}}
               obj2); // expected-warning {{object whose reference is captured does not live long enough}}
  }                              // expected-note 2 {{destroyed here}}
  v.use();                       // expected-note 2 {{later used here}}
}

void lifetimebound_multiple_args_potential(bool cond) {
  MyObj safe;
  View v = safe;
  {
    MyObj obj1;
    if (cond) {
      MyObj obj2;
      v = Choose(true,
                 obj1,             // expected-warning {{object whose reference is captured does not live long enough}}
                 obj2);            // expected-warning {{object whose reference is captured does not live long enough}}
    }                              // expected-note {{destroyed here}}
  }                                // expected-note {{destroyed here}}
  v.use();                         // expected-note 2 {{later used here}}
}

View SelectFirst(View a [[clang::lifetimebound]], View b);
void lifetimebound_mixed_args() {
  View v;
  {
    MyObj obj1, obj2;
    v = SelectFirst(obj1,        // expected-warning {{object whose reference is captured does not live long enough}}
                    obj2);
  }                              // expected-note {{destroyed here}}
  v.use();                       // expected-note {{later used here}}
}

struct LifetimeBoundMember {
  LifetimeBoundMember();
  View get() const [[clang::lifetimebound]];
  operator View() const [[clang::lifetimebound]];
};

void lifetimebound_member_function() {
  View v;
  {
    MyObj obj;
    v  = obj.getView(); // expected-warning {{object whose reference is captured does not live long enough}}
  }                     // expected-note {{destroyed here}}
  v.use();              // expected-note {{later used here}}
}

struct LifetimeBoundConversionView {
  LifetimeBoundConversionView();
  ~LifetimeBoundConversionView();
  operator View() const [[clang::lifetimebound]];
};

void lifetimebound_conversion_operator() {
  View v;
  {
    LifetimeBoundConversionView obj;
    v = obj;  // expected-warning {{object whose reference is captured does not live long enough}}
  }           // expected-note {{destroyed here}}
  v.use();    // expected-note {{later used here}}
}

void lifetimebound_chained_calls() {
  View v;
  {
    MyObj obj;
    v = Identity(Identity(Identity(obj))); // expected-warning {{object whose reference is captured does not live long enough}}
  }                                        // expected-note {{destroyed here}}
  v.use();                                 // expected-note {{later used here}}
}

void lifetimebound_with_pointers() {
  MyObj* ptr;
  {
    MyObj obj;
    ptr = GetPointer(obj); // expected-warning {{object whose reference is captured does not live long enough}}
  }                        // expected-note {{destroyed here}}
  (void)*ptr;              // expected-note {{later used here}}
}

void lifetimebound_no_error_safe_usage() {
  MyObj obj;
  View v1 = Identity(obj);      // No warning - obj lives long enough
  View v2 = Choose(true, v1, Identity(obj)); // No warning - all args are safe
  v2.use();                     // Safe usage
}

void lifetimebound_partial_safety(bool cond) {
  MyObj safe_obj;
  View v = safe_obj;
  
  if (cond) {
    MyObj temp_obj;
    v = Choose(true, 
               safe_obj,
               temp_obj); // expected-warning {{object whose reference is captured does not live long enough}}
  }                       // expected-note {{destroyed here}}
  v.use();                // expected-note {{later used here}}
}

const MyObj& GetObject(View v [[clang::lifetimebound]]);
void lifetimebound_return_reference() {
  View v;
  const MyObj* ptr;
  {
    MyObj obj;
    View temp_v = obj;  // expected-warning {{object whose reference is captured does not live long enough}}
    const MyObj& ref = GetObject(temp_v);
    ptr = &ref;
  }                       // expected-note {{destroyed here}}
  (void)*ptr;             // expected-note {{later used here}}
}

struct LifetimeBoundCtor {
  LifetimeBoundCtor();
  LifetimeBoundCtor(const MyObj& obj [[clang::lifetimebound]]);
};

void lifetimebound_ctor() {
  LifetimeBoundCtor v;
  {
    MyObj obj;
    v = obj; // expected-warning {{object whose reference is captured does not live long enough}}
  }          // expected-note {{destroyed here}}
  (void)v;   // expected-note {{later used here}}
}

View lifetimebound_return_of_local() {
  MyObj stack;
  return Identity(stack); // expected-warning {{address of stack memory is returned later}}
                          // expected-note@-1 {{returned here}}
}

const MyObj& lifetimebound_return_ref_to_local() {
  MyObj stack;
  return IdentityRef(stack); // expected-warning {{address of stack memory is returned later}}
                             // expected-note@-1 {{returned here}}
}

View lifetimebound_return_by_value_param(MyObj stack_param) {
  return Identity(stack_param); // expected-warning {{address of stack memory is returned later}}
                                // expected-note@-1 {{returned here}}
}

View lifetimebound_return_by_value_multiple_param(int cond, MyObj a, MyObj b, MyObj c) {
  if (cond == 1) 
    return Identity(a); // expected-warning {{address of stack memory is returned later}}
                        // expected-note@-1 {{returned here}}
  if (cond == 2) 
    return Identity(b); // expected-warning {{address of stack memory is returned later}}
                        // expected-note@-1 {{returned here}}
  return Identity(c); // expected-warning {{address of stack memory is returned later}}
                      // expected-note@-1 {{returned here}}
}

template<class T>
View lifetimebound_return_by_value_param_template(T t) {
  return Identity(t); // expected-warning {{address of stack memory is returned later}}
                      // expected-note@-1 {{returned here}}
}
void use_lifetimebound_return_by_value_param_template() { 
  lifetimebound_return_by_value_param_template(MyObj{}); // function-note {{in instantiation of}}
}

void lambda_uar_param() {
  auto lambda = [](MyObj stack_param) {
    return Identity(stack_param); // expected-warning {{address of stack memory is returned later}}
                                  // expected-note@-1 {{returned here}}
  };
  lambda(MyObj{});
}

// FIXME: This should be detected. We see correct destructors but origin flow breaks somewhere.
namespace VariadicTemplatedParamsUAR {

template<typename... Args>
View Max(Args... args [[clang::lifetimebound]]);

template<typename... Args>
View lifetimebound_return_of_variadic_param(Args... args) {
  return Max(args...);
}
void test_variadic() {
  lifetimebound_return_of_variadic_param(MyObj{1}, MyObj{2}, MyObj{3});
}
} // namespace VariadicTemplatedParamsUAR

// FIXME: Fails to diagnose UAF when a reference to a by-value param escapes via an out-param.
void uaf_from_by_value_param_failing(MyObj param, View* out_p) {
  *out_p = Identity(param);
}

// Conditional operator.
void conditional_operator_one_unsafe_branch(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  {
    MyObj temp;
    p = cond ? &temp  // expected-warning {{object whose reference is captured does not live long enough}}
             : &safe;
  }  // expected-note {{destroyed here}}

  // This is not a use-after-free for any value of `cond` but the analysis
  // cannot reason this and marks the above as a false positive. This 
  // ensures safety regardless of cond's value.
  if (cond) 
    p = &safe;
  (void)*p;  // expected-note {{later used here}}
}

void conditional_operator_two_unsafe_branches(bool cond) {
  MyObj* p;
  {
    MyObj a, b;
    p = cond ? &a   // expected-warning {{object whose reference is captured does not live long enough}}
             : &b;  // expected-warning {{object whose reference is captured does not live long enough}}
  }  // expected-note 2 {{destroyed here}}
  (void)*p;  // expected-note 2 {{later used here}}
}

void conditional_operator_nested(bool cond) {
  MyObj* p;
  {
    MyObj a, b, c, d;
    p = cond ? cond ? &a    // expected-warning {{object whose reference is captured does not live long enough}}.
                    : &b    // expected-warning {{object whose reference is captured does not live long enough}}.
             : cond ? &c    // expected-warning {{object whose reference is captured does not live long enough}}.
                    : &d;   // expected-warning {{object whose reference is captured does not live long enough}}.
  }  // expected-note 4 {{destroyed here}}
  (void)*p;  // expected-note 4 {{later used here}}
}

void conditional_operator_lifetimebound(bool cond) {
  MyObj* p;
  {
    MyObj a, b;
    p = Identity(cond ? &a    // expected-warning {{object whose reference is captured does not live long enough}}
                      : &b);  // expected-warning {{object whose reference is captured does not live long enough}}
  }  // expected-note 2 {{destroyed here}}
  (void)*p;  // expected-note 2 {{later used here}}
}

void conditional_operator_lifetimebound_nested(bool cond) {
  MyObj* p;
  {
    MyObj a, b;
    p = Identity(cond ? Identity(&a)    // expected-warning {{object whose reference is captured does not live long enough}}
                      : Identity(&b));  // expected-warning {{object whose reference is captured does not live long enough}}
  }  // expected-note 2 {{destroyed here}}
  (void)*p;  // expected-note 2 {{later used here}}
}

void conditional_operator_lifetimebound_nested_deep(bool cond) {
  MyObj* p;
  {
    MyObj a, b, c, d;
    p = Identity(cond ? Identity(cond ? &a     // expected-warning {{object whose reference is captured does not live long enough}}
                                      : &b)    // expected-warning {{object whose reference is captured does not live long enough}}
                      : Identity(cond ? &c     // expected-warning {{object whose reference is captured does not live long enough}}
                                      : &d));  // expected-warning {{object whose reference is captured does not live long enough}}
  }  // expected-note 4 {{destroyed here}}
  (void)*p;  // expected-note 4 {{later used here}}
}

void parentheses(bool cond) {
  MyObj* p;
  {
    MyObj a;
    p = &((((a))));  // expected-warning {{object whose reference is captured does not live long enough}}
  }                  // expected-note {{destroyed here}}
  (void)*p;          // expected-note {{later used here}}

  {
    MyObj a;
    p = ((GetPointer((a))));  // expected-warning {{object whose reference is captured does not live long enough}}
  }                           // expected-note {{destroyed here}}
  (void)*p;                   // expected-note {{later used here}}

  {
    MyObj a, b, c, d;
    p = &(cond ? (cond ? a     // expected-warning {{object whose reference is captured does not live long enough}}.
                       : b)    // expected-warning {{object whose reference is captured does not live long enough}}.
               : (cond ? c     // expected-warning {{object whose reference is captured does not live long enough}}.
                       : d));  // expected-warning {{object whose reference is captured does not live long enough}}.
  }  // expected-note 4 {{destroyed here}}
  (void)*p;  // expected-note 4 {{later used here}}

  {
    MyObj a, b, c, d;
    p = ((cond ? (((cond ? &a : &b)))   // expected-warning 2 {{object whose reference is captured does not live long enough}}.
              : &(((cond ? c : d)))));  // expected-warning 2 {{object whose reference is captured does not live long enough}}.
  }  // expected-note 4 {{destroyed here}}
  (void)*p;  // expected-note 4 {{later used here}}

}

void use_temporary_after_destruction() {
  View a;
  a = non_trivially_destructed_temporary(); // expected-warning {{object whose reference is captured does not live long enough}} \
                  expected-note {{destroyed here}}
  use(a); // expected-note {{later used here}}
}

void passing_temporary_to_lifetime_bound_function() {
  View a = construct_view(non_trivially_destructed_temporary()); // expected-warning {{object whose reference is captured does not live long enough}} \
                expected-note {{destroyed here}}
  use(a); // expected-note {{later used here}}
}

void use_trivial_temporary_after_destruction() {
  View a;
  a = trivially_destructed_temporary(); // expected-warning {{object whose reference is captured does not live long enough}} \
                expected-note {{destroyed here}}
  use(a); // expected-note {{later used here}}
}

namespace GH162834 {
// https://github.com/llvm/llvm-project/issues/162834
template <class T>
struct StatusOr {
  ~StatusOr() {}
  const T& value() const& [[clang::lifetimebound]] { return data; }

  private:
  T data;
};

StatusOr<View> getViewOr();
StatusOr<MyObj> getStringOr();
StatusOr<MyObj*> getPointerOr();

void foo() {
  View view;
  {
    StatusOr<View> view_or = getViewOr();
    view = view_or.value();
  }
  (void)view;
}

void bar() {
  MyObj* pointer;
  {
    StatusOr<MyObj*> pointer_or = getPointerOr();
    pointer = pointer_or.value();
  }
  (void)*pointer;
}

void foobar() {
  View view;
  {
    StatusOr<MyObj> string_or = getStringOr();
    view = string_or. // expected-warning {{object whose reference is captured does not live long enough}}
            value();
  }                     // expected-note {{destroyed here}}
  (void)view;           // expected-note {{later used here}}
}
} // namespace GH162834

namespace RangeBasedForLoop {
struct MyObjStorage {
  MyObj objs[1];
  MyObjStorage() {}
  ~MyObjStorage() {}
  const MyObj *begin() const [[clang::lifetimebound]]  { return objs; }
  const MyObj *end() const { return objs + 1; }
};

void range_based_for_use_after_scope() {
  View v;
  {
    MyObjStorage s;
    for (const MyObj &o : s) { // expected-warning {{object whose reference is captured does not live long enough}}
      v = o;
    }
  } // expected-note {{destroyed here}}
  v.use(); // expected-note {{later used here}}
}

View range_based_for_use_after_return() {
  MyObjStorage s;
  for (const MyObj &o : s) { // expected-warning {{address of stack memory is returned later}}
    return o;  // expected-note {{returned here}}
  }
  return *s.begin();  // expected-warning {{address of stack memory is returned later}}
                      // expected-note@-1 {{returned here}}
}

void range_based_for_not_reference() {
  View v;
  {
    MyObjStorage s;
    for (MyObj o : s) { // expected-note {{destroyed here}}
      v = o; // expected-warning {{object whose reference is captured does not live long enough}}
    }
  }
  v.use();  // expected-note {{later used here}}
}

void range_based_for_no_error() {
  View v;
  MyObjStorage s;
  for (const MyObj &o : s) {
    v = o;
  }
  v.use();
}

} // namespace RangeBaseForLoop

namespace UserDefinedDereference {
// Test user-defined dereference operators with lifetimebound
template<typename T>
struct SmartPtr {
  T* ptr;
  SmartPtr() {}
  SmartPtr(T* p) : ptr(p) {}
  T& operator*() const [[clang::lifetimebound]] { return *ptr; }
  T* operator->() const [[clang::lifetimebound]] { return ptr; }
};

void test_user_defined_deref_uaf() {
  MyObj* p;
  {
    MyObj obj;
    SmartPtr<MyObj> smart_ptr(&obj);
    p = &(*smart_ptr);  // expected-warning {{object whose reference is captured does not live long enough}}
  }                     // expected-note {{destroyed here}}
  (void)*p;             // expected-note {{later used here}}
}

MyObj& test_user_defined_deref_uar() {
  MyObj obj;
  SmartPtr<MyObj> smart_ptr(&obj);
  return *smart_ptr;  // expected-warning {{address of stack memory is returned later}}
                      // expected-note@-1 {{returned here}}
}

void test_user_defined_deref_with_view() {
  View v;
  {
    MyObj obj;
    SmartPtr<MyObj> smart_ptr(&obj);
    v = *smart_ptr;  // expected-warning {{object whose reference is captured does not live long enough}}
  }                  // expected-note {{destroyed here}}
  v.use();           // expected-note {{later used here}}
}

void test_user_defined_deref_arrow() {
  MyObj* p;
  {
    MyObj obj;
    SmartPtr<MyObj> smart_ptr(&obj);
    p = smart_ptr.operator->();  // expected-warning {{object whose reference is captured does not live long enough}}
  }                              // expected-note {{destroyed here}}
  (void)*p;                      // expected-note {{later used here}}
}

void test_user_defined_deref_chained() {
  MyObj* p;
  {
    MyObj obj;
    SmartPtr<SmartPtr<MyObj>> double_ptr;
    p = &(**double_ptr);  // expected-warning {{object whose reference is captured does not live long enough}}
  }                       // expected-note {{destroyed here}}
  (void)*p;               // expected-note {{later used here}}
}

} // namespace UserDefinedDereference

namespace structured_binding {
struct Pair {
  MyObj a;
  MyObj b;
  Pair() {}
  ~Pair() {}
};

// FIXME: Detect this.
void structured_binding_use_after_scope() {
  View v;
  {
    Pair p;
    auto &[a_ref, b_ref] = p;
    v = a_ref;
  }
  v.use();
}
}

namespace MaxFnLifetimeBound {

template<class T>
T&& MaxT(T&& a [[clang::lifetimebound]], T&& b [[clang::lifetimebound]]);

const MyObj& call_max_with_obj() {
  MyObj oa, ob;
  return  MaxT(oa,    // expected-warning {{address of stack memory is returned later}}          
                      // expected-note@-1 2 {{returned here}}
               ob);   // expected-warning {{address of stack memory is returned later}}
                    
}

MyObj* call_max_with_obj_error() {
  MyObj oa, ob;
  return  &MaxT(oa,   // expected-warning {{address of stack memory is returned later}}          
                      // expected-note@-1 2 {{returned here}}
                ob);  // expected-warning {{address of stack memory is returned later}}
}

const MyObj* call_max_with_ref_obj_error() {
  MyObj oa, ob;
  const MyObj& refa = oa;     // expected-warning {{address of stack memory is returned later}}
  const MyObj& refb = ob;     // expected-warning {{address of stack memory is returned later}}
  return  &MaxT(refa, refb);  // expected-note 2 {{returned here}}
}
const MyObj& call_max_with_ref_obj_return_ref_error() {
  MyObj oa, ob;
  const MyObj& refa = oa;     // expected-warning {{address of stack memory is returned later}}
  const MyObj& refb = ob;     // expected-warning {{address of stack memory is returned later}}
  return  MaxT(refa, refb);   // expected-note 2 {{returned here}}
}

MyObj call_max_with_obj_no_error() {
  MyObj oa, ob;
  return  MaxT(oa, ob);
}

const MyObj& call_max_with_ref_obj_no_error(const MyObj& a, const MyObj& b) {
  return  MaxT(a, b);
}

const View& call_max_with_view_with_error() {
  View va, vb;
  return MaxT(va,   // expected-warning {{address of stack memory is returned later}}
                    // expected-note@-1 2 {{returned here}}
              vb);  // expected-warning {{address of stack memory is returned later}}
}

struct [[gsl::Pointer]] NonTrivialPointer  { ~NonTrivialPointer(); };

const NonTrivialPointer& call_max_with_non_trivial_view_with_error() {
  NonTrivialPointer va, vb;
  return MaxT(va,   // expected-warning {{address of stack memory is returned later}}
                    // expected-note@-1 2 {{returned here}}
              vb);  // expected-warning {{address of stack memory is returned later}}
}

namespace MultiPointerTypes {
int** return_2p() {
  int a = 1;
  int* b = &a;  // expected-warning {{address of stack memory is returned later}}
  int** c = &b; // expected-warning {{address of stack memory is returned later}}
  return c;     // expected-note 2 {{returned here}}
}

int** return_2p_one_is_safe(int& a) {
  int* b = &a;
  int** c = &b; // expected-warning {{address of stack memory is returned later}}
  return c;     // expected-note {{returned here}}
}

int*** return_3p() {
  int a = 1;
  int* b = &a;    // expected-warning {{address of stack memory is returned later}}
  int** c = &b;   // expected-warning {{address of stack memory is returned later}}
  int*** d = &c;  // expected-warning {{address of stack memory is returned later}}
  return d;       // expected-note 3 {{returned here}}
}

View** return_view_p() {
  MyObj a;
  View b = a;     // expected-warning {{address of stack memory is returned later}}
  View* c = &b;   // expected-warning {{address of stack memory is returned later}}
  View** d = &c;  // expected-warning {{address of stack memory is returned later}}
  return d;       // expected-note 3 {{returned here}}
}

} // namespace MultiPointerTypes

View call_max_with_view_without_error() {
  View va, vb;
  return MaxT(va, vb);
}

} // namespace StdMaxStyleLifetimeBound

namespace CppCoverage {

int getInt();

void ReferenceParam(unsigned Value, unsigned &Ref) {
  Value = getInt();
  Ref = getInt();
}

inline void normalize(int &exponent, int &mantissa) {
  const int shift = 1;
  exponent -= shift;
  mantissa <<= shift;
}

void add(int c, MyObj* node) {
  MyObj* arr[10];
  arr[4] = node;
}
} // namespace CppCoverage

namespace strict_warn_on_move {
void strict_warn_on_move() {
  MyObj b;
  View v;
  {
    MyObj a;
    v = a;            // expected-warning-re {{object whose reference {{.*}} may have been moved}}
    b = std::move(a); // expected-note {{potentially moved here}}
  }                   // expected-note {{destroyed here}}
  (void)v;            // expected-note {{later used here}}
}

void flow_sensitive(bool c) {
  View v;
  {
    MyObj a;
    if (c) {
      MyObj b = std::move(a);
      return;
    }
    v = a;  // expected-warning {{object whose reference}}
  }         // expected-note {{destroyed here}}
  (void)v;  // expected-note {{later used here}}
}

void take(MyObj&&);
void detect_conditional(bool cond) {
  View v;
  {
    MyObj a, b;
    v = cond ? a : b; // expected-warning-re 2 {{object whose reference {{.*}} may have been moved}}
    take(std::move(cond ? a : b)); // expected-note 2 {{potentially moved here}}
  }         // expected-note 2 {{destroyed here}}
  (void)v;  // expected-note 2 {{later used here}}
}

void wrong_use_of_move_is_permissive() {
  View v;
  {
    MyObj a;
    v = std::move(a); // expected-warning {{object whose reference is captured does not live long enough}}
  }         // expected-note {{destroyed here}}
  (void)v;  // expected-note {{later used here}}
  const int* p;
  {
    MyObj a;
    p = std::move(a).getData(); // expected-warning {{object whose reference is captured does not live long enough}}
  }         // expected-note {{destroyed here}}
  (void)p;  // expected-note {{later used here}}
}

void take(int*);
void test_release_no_uaf() {
  int* r;
  // Calling release() marks p as moved from, so its destruction doesn't invalidate r.
  {
    std::unique_ptr<int> p;
    r = p.get();        // expected-warning-re {{object whose reference {{.*}} may have been moved}}
    take(p.release());  // expected-note {{potentially moved here}}
  }                     // expected-note {{destroyed here}}
  (void)*r;             // expected-note {{later used here}}
}
} // namespace strict_warn_on_move

// Implicit this annotations with redecls.
namespace GH172013 {
// https://github.com/llvm/llvm-project/issues/62072
// https://github.com/llvm/llvm-project/issues/172013
struct S {
    View x() const [[clang::lifetimebound]];
    MyObj i;
};

View S::x() const { return i; }

void bar() {
    View x;
    {
        S s;
        x = s.x(); // expected-warning {{object whose reference is captured does not live long enough}}
        View y = S().x(); // expected-warning {{object whose reference is captured does not live long enough}} \
                             expected-note {{destroyed here}}
        (void)y; // expected-note {{used here}}
    } // expected-note {{destroyed here}}
    (void)x; // expected-note {{used here}}
}
}

namespace DereferenceViews {
const MyObj& testDeref(MyObj obj) {
  View v = obj; // expected-warning {{address of stack memory is returned later}}
  return *v;    // expected-note {{returned here}}
}
const MyObj* testDerefAddr(MyObj obj) {
  View v = obj; // expected-warning {{address of stack memory is returned later}}
  return &*v;   // expected-note {{returned here}}
}
const MyObj* testData(MyObj obj) {
  View v = obj;     // expected-warning {{address of stack memory is returned later}}
  return v.data();  // expected-note {{returned here}}
}
const int* testLifetimeboundAccessorOfMyObj(MyObj obj) {
  View v = obj;           // expected-warning {{address of stack memory is returned later}}
  const MyObj* ptr = v.data();
  return ptr->getData();  // expected-note {{returned here}}
}
const int* testLifetimeboundAccessorOfMyObjThroughDeref(MyObj obj) {
  View v = obj;         // expected-warning {{address of stack memory is returned later}}
  return v->getData();  // expected-note {{returned here}}
}
} // namespace DereferenceViews

namespace ViewsBeginEndIterators {
template <typename T>
struct [[gsl::Pointer]] Iterator {
  Iterator operator++();
  T& operator*() const;
  T* operator->() const;
  bool operator!=(const Iterator& other) const;
};

template <typename T>
struct [[gsl::Owner]] Container {
using It = Iterator<T>;
It begin() const [[clang::lifetimebound]];
It end() const [[clang::lifetimebound]];
};

MyObj Global;

const MyObj& ContainerMyObjReturnRef(Container<MyObj> c) {
  for (const MyObj& x : c) {  // expected-warning {{address of stack memory is returned later}}
    return x;                 // expected-note {{returned here}}
  }
  return Global;
}

View ContainerMyObjReturnView(Container<MyObj> c) {
  for (const MyObj& x : c) {  // expected-warning {{address of stack memory is returned later}}
    return x;                 // expected-note {{returned here}}
  }
  for (View x : c) {  // expected-warning {{address of stack memory is returned later}}
    return x;         // expected-note {{returned here}}
  }
  return Global;
}

View ContainerViewsOk(Container<View> c) {
  for (View x : c) {
    return x;
  }
  for (const View& x : c) {
    return x;
  }
  return Global;
}
} // namespace ViewsBeginEndIterators

namespace reference_type_decl_ref_expr {
struct S {
  S();
  ~S();
  const std::string& x() const [[clang::lifetimebound]];
};

const std::string& identity(const std::string& in [[clang::lifetimebound]]);
const S& identity(const S& in [[clang::lifetimebound]]);

void test_temporary() {
  const std::string& x = S().x(); // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}
  (void)x; // expected-note {{later used here}}

  const std::string& y = identity(S().x()); // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}
  (void)y; // expected-note {{later used here}}

  std::string_view z;
  {
    S s;
    const std::string& zz = s.x(); // expected-warning {{object whose reference is captured does not live long enough}}
    z = zz;
  } // expected-note {{destroyed here}}
  (void)z; // expected-note {{later used here}}
}

void test_lifetime_extension_ok() {
  const S& x = S();
  (void)x;
  const S& y = identity(S()); // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}
  (void)y; // expected-note {{later used here}}
}

const std::string& test_return() {
  const std::string& x = S().x(); // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}
  return x; // expected-note {{later used here}}
}
} // namespace reference_type_decl_ref_expr

namespace field_access {

struct S {
  std::string s;
  std::string_view sv;
};

void uaf() {
  std::string_view view;
  {
    S str;
    S* p = &str;  // expected-warning {{object whose reference is captured does not live long enough}}
    view = p->s;
  } // expected-note {{destroyed here}}
  (void)view;  // expected-note {{later used here}}
}

void not_uaf() {
  std::string_view view;
  {
    S str;
    S* p = &str;
    view = p->sv;
  }
  (void)view;
}

union U {
  std::string s;
  std::string_view sv;
  ~U() {}
};

void uaf_union() {
  std::string_view view;
  {
    U u = U{"hello"};
    U* up = &u;  // expected-warning {{object whose reference is captured does not live long enough}}
    view = up->s;
  } // expected-note {{destroyed here}}
  (void)view;  // expected-note {{later used here}}
}

struct AnonymousUnion {
union {
  int x;
  float y;
};
};

void uaf_anonymous_union() {
  int* ip;
  {
    AnonymousUnion au;
    AnonymousUnion* up = &au;  // expected-warning {{object whose reference is captured does not live long enough}}
    ip = &up->x;
  } // expected-note {{destroyed here}}
  (void)ip;  // expected-note {{later used here}}
}

struct RefMember {
  std::string& str_ref;
  std::string* str_ptr;
  std::string str;
  std::string_view view;
  std::string_view& view_ref;
  RefMember();
  ~RefMember();
};

std::string_view refMemberReturnView1(RefMember a) { return a.str_ref; }
std::string_view refMemberReturnView2(RefMember a) { return *a.str_ptr; }
std::string_view refMemberReturnView3(RefMember a) { return a.str; } // expected-warning {{address of stack memory is returned later}} expected-note {{returned here}}
std::string& refMemberReturnRef1(RefMember a) { return a.str_ref; }
std::string& refMemberReturnRef2(RefMember a) { return *a.str_ptr; }
std::string& refMemberReturnRef3(RefMember a) { return a.str; } // expected-warning {{address of stack memory is returned later}} expected-note {{returned here}}
std::string_view refViewMemberReturnView1(RefMember a) { return a.view; }
std::string_view& refViewMemberReturnView2(RefMember a) { return a.view; } // expected-warning {{address of stack memory is returned later}} expected-note {{returned here}}
std::string_view refViewMemberReturnRefView1(RefMember a) { return a.view_ref; }
std::string_view& refViewMemberReturnRefView2(RefMember a) { return a.view_ref; }
} // namespace field_access

namespace attr_on_template_params {
struct MyObj {
  ~MyObj();
};

template <typename T>
struct MemberFuncsTpl {
  ~MemberFuncsTpl();
  // Template Version A: Attribute on declaration only
  const T* memberA(const T& x [[clang::lifetimebound]]);
  // Template Version B: Attribute on definition only
  const T* memberB(const T& x);
  // Template Version C: Attribute on BOTH declaration and definition
  const T* memberC(const T& x [[clang::lifetimebound]]);
};

template <typename T>
const T* MemberFuncsTpl<T>::memberA(const T& x) {
    return &x;
}
template <typename T>
const T* MemberFuncsTpl<T>::memberB(const T& x [[clang::lifetimebound]]) {
    return &x;
}
template <typename T>
const T* MemberFuncsTpl<T>::memberC(const T& x [[clang::lifetimebound]]) {
    return &x;
}

void test() {
  MemberFuncsTpl<MyObj> mtf;
  const MyObj* pTMA = mtf.memberA(MyObj()); // expected-warning {{object whose reference is captured does not live long enough}} // expected-note {{destroyed here}}
  const MyObj* pTMB = mtf.memberB(MyObj()); // tu-warning {{object whose reference is captured does not live long enough}} // tu-note {{destroyed here}}
  const MyObj* pTMC = mtf.memberC(MyObj()); // expected-warning {{object whose reference is captured does not live long enough}} // expected-note {{destroyed here}}
  (void)pTMA; // expected-note {{later used here}}
  (void)pTMB; // tu-note {{later used here}}
  (void)pTMC; // expected-note {{later used here}}
}

} // namespace attr_on_template_params

namespace non_trivial_views {
struct [[gsl::Pointer]] View {
    View(const std::string&);
    ~View(); // Forces a CXXBindTemporaryExpr.
};

View test1(std::string a) {
  // Make sure we handle CXXBindTemporaryExpr of view types.
  return View(a); // expected-warning {{address of stack memory is returned later}} expected-note {{returned here}}
}

View test2(std::string a) {
  View b = View(a); // expected-warning {{address of stack memory is returned later}}
  return b;         // expected-note {{returned here}}
}

View test3(std::string a) {
  const View& b = View(a);  // expected-warning {{address of stack memory is returned later}}
  return b;                 // expected-note {{returned here}}
}
} // namespace non_trivial_views

namespace OwnerArrowOperator {
void test_optional_arrow() {
  const char* p;
  {
    std::optional<std::string> opt;
    p = opt->data();  // expected-warning {{object whose reference is captured does not live long enough}}
  }                   // expected-note {{destroyed here}}
  (void)*p;           // expected-note {{later used here}}
}

void test_optional_arrow_lifetimebound() {
  View v;
  {
    std::optional<MyObj> opt;
    v = opt->getView();  // expected-warning {{object whose reference is captured does not live long enough}}
  }                      // expected-note {{destroyed here}}
  v.use();               // expected-note {{later used here}}
}

void test_unique_ptr_arrow() {
  const char* p;
  {
    std::unique_ptr<std::string> up;
    p = up->data();  // expected-warning {{object whose reference is captured does not live long enough}}
  }                  // expected-note {{destroyed here}}
  (void)*p;          // expected-note {{later used here}}
}

void test_optional_view_arrow() {
    const char* p;
    {
        std::optional<std::string_view> opt;
        p = opt->data();
    }
    (void)*p;
}
} // namespace OwnerArrowOperator

namespace lambda_captures {
auto return_ref_capture() {
  int local = 1;
  auto lambda = [&local]() { return local; }; // expected-warning {{address of stack memory is returned later}}
  return lambda; // expected-note {{returned here}}
}

void safe_ref_capture() {
  int local = 1;
  auto lambda = [&local]() { return local; };
  lambda();
}

auto capture_int_by_value() {
  int x = 1;
  auto lambda = [x]() { return x; };
  return lambda;
}

auto capture_view_by_value() {
  MyObj obj;
  View v(obj); // expected-warning {{address of stack memory is returned later}}
  auto lambda = [v]() { return v; };
  return lambda; // expected-note {{returned here}}
}

void capture_view_by_value_safe() {
  MyObj obj;
  View v(obj);
  auto lambda = [v]() { return v; };
  lambda();
}

auto capture_pointer_by_ref() {
  MyObj obj;
  MyObj* p = &obj;
  auto lambda = [&p]() { return p; }; // expected-warning {{address of stack memory is returned later}}
  return lambda; // expected-note {{returned here}}
}

auto capture_multiple() {
  int a, b;
  auto lambda = [
    &a,  // expected-warning {{address of stack memory is returned later}}
    &b   // expected-warning {{address of stack memory is returned later}}
  ]() { return a + b; };
  return lambda; // expected-note 2 {{returned here}}
}

auto capture_raw_pointer_by_value() {
  int x;
  int* p = &x; // expected-warning {{address of stack memory is returned later}}
  auto lambda = [p]() { return p; };
  return lambda; // expected-note {{returned here}}
}

auto capture_raw_pointer_init_capture() {
  int x;
  int* p = &x; // expected-warning {{address of stack memory is returned later}}
  auto lambda = [q = p]() { return q; };
  return lambda; // expected-note {{returned here}}
}

auto capture_view_init_capture() {
  MyObj obj;
  View v(obj); // expected-warning {{address of stack memory is returned later}}
  auto lambda = [w = v]() { return w; };
  return lambda; // expected-note {{returned here}}
}

auto capture_lambda() {
  int x;
  auto inner = [&x]() { return x; }; // expected-warning {{address of stack memory is returned later}}
  auto outer = [inner]() { return inner(); };
  return outer; // expected-note {{returned here}}
}

auto return_copied_lambda() {
  int local = 1;
  auto lambda = [&local]() { return local; }; // expected-warning {{address of stack memory is returned later}}
  auto lambda_copy = lambda;
  return lambda_copy; // expected-note {{returned here}}
}

auto implicit_ref_capture() {
  int local = 1;
  auto lambda = [&]() { return local; }; // expected-warning {{address of stack memory is returned later}}
  return lambda; // expected-note {{returned here}}
}

// TODO: Include the name of the variable in the diagnostic to improve
// clarity, especially for implicit lambda captures where multiple warnings
// can point to the same source location.
auto implicit_ref_capture_multiple() {
  int local = 1, local2 = 2;
  auto lambda = [&]() { return local + local2; }; // expected-warning 2 {{address of stack memory is returned later}}
  return lambda; // expected-note 2 {{returned here}}
}

auto implicit_value_capture() {
  MyObj obj;
  View v(obj); // expected-warning {{address of stack memory is returned later}}
  auto lambda = [=]() { return v; };
  return lambda; // expected-note {{returned here}}
}

auto* pointer_to_lambda_outlives() {
  auto lambda = []() { return 42; };
  return &lambda; // expected-warning {{address of stack memory is returned later}} \
                  // expected-note {{returned here}}
}

auto capture_static() {
  static int local = 1;
  // Only automatic storage duration variables may be captured.
  // Variables with static storage duration behave like globals and are directly accessible.
  // The below lambdas should not capture `local`.
  auto lambda = [&]() { return local; };
  auto lambda2 = []() { return local; };
  lambda2();
  return lambda;
}

auto capture_static_address_by_value() {
  static int local = 1;
  int* p = &local;
  auto lambda = [p]() { return p; };
  return lambda;
}

auto capture_static_address_by_ref() {
  static int local = 1;
  int* p = &local;
  auto lambda = [&p]() { return p; }; // expected-warning {{address of stack memory is returned later}}
  return lambda; // expected-note {{returned here}}
}

auto capture_multilevel_pointer() {
  int x;
  int *p = &x; // expected-warning {{address of stack memory is returned later}}
  int **q = &p; // expected-warning {{address of stack memory is returned later}}
  int ***r = &q; // expected-warning {{address of stack memory is returned later}}
  auto lambda = [=]() { return *p + **q + ***r; };
  return lambda; // expected-note 3 {{returned here}}
}
} // namespace lambda_captures

namespace LoopLocalPointers {

void conditional_assignment_in_loop() {
  for (int i = 0; i < 10; ++i) {
    MyObj obj;
    MyObj* view;
    if (i > 5) {
      view = &obj;
    }
    (void)*view;
  }
}

void unconditional_assignment_in_loop() {
  for (int i = 0; i < 10; ++i) {
    MyObj obj;
    MyObj* view = &obj;
    (void)*view;
  }
}

// FIXME: False positive. Requires modeling flow-sensitive aliased origins
// to properly expire pp's inner origin when p's lifetime ends.
void multi_level_pointer_in_loop() {
  for (int i = 0; i < 10; ++i) {
    MyObj obj;
    MyObj* p;
    MyObj** pp;
    if (i > 5) {
      p = &obj; // expected-warning {{object whose reference is captured does not live long enough}}
      pp = &p;
    }
    (void)**pp; // expected-note {{later used here}}
  }             // expected-note {{destroyed here}}
}

void outer_pointer_outlives_inner_pointee() {
  MyObj safe;
  MyObj* view = &safe;
  for (int i = 0; i < 10; ++i) {
    MyObj obj;
    view = &obj;     // expected-warning {{object whose reference is captured does not live long enough}}
  }                  // expected-note {{destroyed here}}
  (void)*view;       // expected-note {{later used here}}
}

} // namespace LoopLocalPointers

namespace array {

void element_use_after_scope() {
  int* p;
  {
    int a[10]{};
    p = &a[2]; // expected-warning {{object whose reference is captured does not live long enough}}
  }            // expected-note {{destroyed here}}
  (void)*p;    // expected-note {{later used here}}
}

int* element_use_after_return() {
  int a[10]{};
  int* p = &a[0]; // expected-warning {{address of stack memory is returned later}}
  return p;       // expected-note {{returned here}}
}

void element_use_same_scope() {
  int a[10]{};
  int* p = &a[0];
  (void)*p;
}

void element_reassigned_safe() {
  int safe[10]{};
  int* p;
  {
    int a[10]{};
    p = &a[0];
  }
  p = &safe[0]; // Rescued.
  (void)*p;
}

void multidimensional_use_after_scope() {
  int* p;
  {
    int a[3][4]{};
    p = &a[1][2]; // expected-warning {{object whose reference is captured does not live long enough}}
  }               // expected-note {{destroyed here}}
  (void)*p;       // expected-note {{later used here}}
}

void member_array_element_use_after_scope() {
  struct S {
    int arr[10];
    int b;
  };
  int* p;
  {
    S s;
    p = &s.arr[0]; // expected-warning {{object whose reference is captured does not live long enough}}
  }                // expected-note {{destroyed here}}
  (void)*p;        // expected-note {{later used here}}
}

void array_of_pointers_use_after_scope() {
  int** p;
  {
    int* a[10]{};
    p = a;  // expected-warning {{object whose reference is captured does not live long enough}}
  }         // expected-note {{destroyed here}}
  (void)*p; // expected-note {{later used here}}
}

void reversed_subscript_use_after_scope() {
  int* p;
  {
    int a[10]{};
    p = &(0[a]); // expected-warning {{object whose reference is captured does not live long enough}}
  }              // expected-note {{destroyed here}}
  (void)*p;      // expected-note {{later used here}}
}

int* return_decayed_array() {
  int a[10]{};
  int *p = a; // expected-warning {{address of stack memory is returned later}}
  return p;   // expected-note {{returned here}}
}

int* param_array_element(int a[], int n) {
  return &a[n];
}

int* static_array() {
  static int a[10]{};
  return &a[1];
}

void pointer_arithmetic_use_after_scope() {
  int* p;
  int* p2;
  int* p3;
  {
    int a[10]{};
    p = a + 5;  // expected-warning {{object whose reference is captured does not live long enough}}
    p2 = a - 5; // expected-warning {{object whose reference is captured does not live long enough}}
    p3 = 5 + a; // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note 3 {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
  (void)*p2;    // expected-note {{later used here}}
  (void)*p3;    // expected-note {{later used here}}
}

// FIXME: Copying a pointer value out of an array element is not tracked.
void copy_pointer_from_array_use_after_scope() {
  int* q;
  {
    int x = 0;
    int* arr[10] = {&x};
    q = arr[0];
  }
  (void)*q; // Should warn.
}

// FIXME: A pointer inside an array becoming dangling is not detected.
void pointer_in_array_use_after_scope() {
  int* arr[10];
  {
    int x = 0;
    arr[0] = &x;
  }
  (void)*arr[0]; // Should warn.
}

} // namespace array

namespace static_call_operator {
// https://github.com/llvm/llvm-project/issues/187426

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++23-extensions"

struct S {
  static S operator()(int, int&&);
  static S& operator()(std::string&&,
                       const int& a [[clang::lifetimebound]],
                       const int& b [[clang::lifetimebound]]);
};

void indexing_with_static_operator() {
  S()(1, 2);
  S& x = S()("1",
             2,  // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}
             3); // expected-warning {{object whose reference is captured does not live long enough}} expected-note {{destroyed here}}

  (void)x; // expected-note 2 {{later used here}}

}
} // namespace static_call_operator

namespace track_origins_for_lifetimebound_record_type {

template <class T> void use(T);

struct S {
  S();
  S(const std::string &s [[clang::lifetimebound]]);

  S return_self_after_registration() const;
  std::string_view getData() const [[clang::lifetimebound]];
};

S getS(const std::string &s [[clang::lifetimebound]]);

void from_free_function() {
  S s = getS(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                   // expected-note {{destroyed here}}
  use(s);                          // expected-note {{later used here}}
}

void from_constructor() {
  S s(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                            // expected-note {{destroyed here}}
  use(s);                   // expected-note {{later used here}}
}

struct Factory {
  S make(const std::string &s [[clang::lifetimebound]]);
  static S create(const std::string &s [[clang::lifetimebound]]);
  S makeThis() const [[clang::lifetimebound]];
};

void from_method() {
  Factory f;
  S s = f.make(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                     // expected-note {{destroyed here}}
  use(s);                            // expected-note {{later used here}}
}

void from_static_method() {
  S s = Factory::create(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                              // expected-note {{destroyed here}}
  use(s);                                     // expected-note {{later used here}}
}

void from_lifetimebound_this_method() {
  S value;
  {
    Factory f;
    value = f.makeThis(); // expected-warning {{object whose reference is captured does not live long enough}}
  }                       // expected-note {{destroyed here}}
  use(value);             // expected-note {{later used here}}
}

void across_scope() {
  S s{};
  {
    std::string str{"abc"};
    s = getS(str); // expected-warning {{object whose reference is captured does not live long enough}}
  }                // expected-note {{destroyed here}}
  use(s);          // expected-note {{later used here}}
}

void same_scope() {
  std::string str{"abc"};
  S s = getS(str);
  use(s);
}

S copy_propagation() {
  std::string str{"abc"};
  S a = getS(str); // expected-warning {{address of stack memory is returned later}}
  S b = a;
  return b; // expected-note {{returned here}}
}

void assignment_propagation() {
  S a, b;
  {
    std::string str{"abc"};
    a = getS(str); // expected-warning {{object whose reference is captured does not live long enough}}
    b = a;
  }                // expected-note {{destroyed here}}
  use(b);          // expected-note {{later used here}}
}

S getSNoAnnotation(const std::string &s);

void no_annotation() {
  S s = getSNoAnnotation(std::string("temp"));
  use(s);
}

void mix_annotated_and_not() {
  S s1 = getS(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                    // expected-note {{destroyed here}}
  S s2 = getSNoAnnotation(std::string("temp"));
  use(s1); // expected-note {{later used here}}
  use(s2);
}

S getS2(const std::string &a [[clang::lifetimebound]], const std::string &b [[clang::lifetimebound]]);

S multiple_lifetimebound_params() {
  std::string str{"abc"};
  S s = getS2(str, std::string("temp")); // expected-warning {{address of stack memory is returned later}} \
                                         // expected-warning {{object whose reference is captured does not live long enough}} \
                                         // expected-note {{destroyed here}}
  return s;                              // expected-note {{returned here}} \
                                         // expected-note {{later used here}}
}

// TODO: Diagnose [[clang::lifetimebound]] on functions whose return value
// cannot refer to any object (e.g., returning int or enum).
int getInt(const std::string &s [[clang::lifetimebound]]);

void primitive_return() {
  int i = getInt(std::string("temp"));
  use(i);
}

template <class T>
T make(const std::string &s [[clang::lifetimebound]]);

void from_template_instantiation() {
  S s = make<S>(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                      // expected-note {{destroyed here}}
  use(s);                             // expected-note {{later used here}}
}

struct FieldInitFromLifetimebound {
  S value; // function-note {{this field dangles}}
  FieldInitFromLifetimebound() : value(getS(std::string("temp"))) {} // function-warning {{address of stack memory escapes to a field}}
};

S S::return_self_after_registration() const {
  std::string s{"abc"};
  getS(s);
  return *this;
}

struct SWithUserDefinedCopyLikeOps {
  SWithUserDefinedCopyLikeOps();
  SWithUserDefinedCopyLikeOps(const std::string &s [[clang::lifetimebound]]) : owned(s), data(s) {}

  SWithUserDefinedCopyLikeOps(const SWithUserDefinedCopyLikeOps &other) : owned("copy"), data(owned) {}

  SWithUserDefinedCopyLikeOps &operator=(const SWithUserDefinedCopyLikeOps &) {
    owned = "copy";
    data = owned;
    return *this;
  }

  std::string owned;
  std::string_view data;
};

SWithUserDefinedCopyLikeOps getSWithUserDefinedCopyLikeOps(const std::string &s [[clang::lifetimebound]]);

SWithUserDefinedCopyLikeOps user_defined_copy_ctor_should_not_assume_origin_propagation() {
  std::string str{"abc"};
  SWithUserDefinedCopyLikeOps s = getSWithUserDefinedCopyLikeOps(str);
  SWithUserDefinedCopyLikeOps copy = s; // Copy is rescued by user-defined copy constructor, so should not warn.
  return copy;
}

void user_defined_assignment_should_not_assume_origin_propagation() {
  SWithUserDefinedCopyLikeOps dst;
  {
    std::string str{"abc"};
    SWithUserDefinedCopyLikeOps src = getSWithUserDefinedCopyLikeOps(str);
    dst = src;
  }
  use(dst);
}

const S &getRef(const std::string &s [[clang::lifetimebound]]);

S from_ref() {
  std::string str{"abc"};
  S s = getRef(str);
  return s;
}

using SAlias = S;
SAlias getSAlias(const std::string &s [[clang::lifetimebound]]);

void from_typedef_return() {
  SAlias s = getSAlias(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                             // expected-note {{destroyed here}}
  use(s);                                    // expected-note {{later used here}}
}

struct SWithOriginPropagatingCopy {
  SWithOriginPropagatingCopy();
  SWithOriginPropagatingCopy(const std::string &s [[clang::lifetimebound]]) : data(s) {}
  SWithOriginPropagatingCopy(const SWithOriginPropagatingCopy &other) : data(other.data) {}
  std::string_view data;
};

SWithOriginPropagatingCopy getSWithOriginPropagatingCopy(const std::string &s [[clang::lifetimebound]]);

// FIXME: False negative. User-defined copy ctor may propagate origins.
SWithOriginPropagatingCopy user_defined_copy_with_origin_propagation() {
  std::string str{"abc"};
  SWithOriginPropagatingCopy s = getSWithOriginPropagatingCopy(str);
  SWithOriginPropagatingCopy copy = s;
  return copy; // Should warn.
}

struct DefaultedOuter {
  DefaultedOuter();
  DefaultedOuter(const std::string &s [[clang::lifetimebound]]) : inner(s) {}
  SWithUserDefinedCopyLikeOps inner;
};

DefaultedOuter getDefaultedOuter(const std::string &s [[clang::lifetimebound]]);

// The defaulted outer copy ctor propagates origins even though the inner
// user-defined copy ctor may break the borrow. This is intentional: this
// pattern does not fit the ownership model this analysis supports.
DefaultedOuter nested_defaulted_outer_with_user_defined_inner() {
  std::string str{"abc"};
  DefaultedOuter o = getDefaultedOuter(str); // expected-warning {{address of stack memory is returned later}}
  DefaultedOuter copy = o;
  return copy; // expected-note {{returned here}}
}

std::string_view getSV(S s [[clang::lifetimebound]]);

// FIXME: False negative. Non-pointer/ref/gsl::Pointer parameter types marked
// [[clang::lifetimebound]] are not registered for origin tracking.
void dangling_view_from_non_pointer_param() {
  std::string_view sv;
  {
    S s;
    sv = getSV(s);
  }
  use(sv); // Should warn.
}

MyObj getMyObj(const MyObj &obj [[clang::lifetimebound]]);

void gsl_owner_return() {
  MyObj obj;
  View v = obj;
  getMyObj(obj);
  use(v);
}

std::vector<std::string_view> createViews(const std::string &s [[clang::lifetimebound]]);

std::span<std::string_view> owner_to_pointer_via_gsl_construction() {
  std::string local;
  auto views = createViews(local);
  return views; // expected-warning {{address of stack memory is returned later}} \
                // expected-note {{returned here}}
}

std::unique_ptr<S> getUniqueS(const std::string &s [[clang::lifetimebound]]);

void owner_return_unique_ptr_s() {
  auto ptr = getUniqueS(std::string("temp")); // expected-warning {{object whose reference is captured does not live long enough}} \
                                              // expected-note {{destroyed here}}
  (void)ptr;                                  // expected-note {{later used here}}
}

std::string_view return_dangling_view_through_owner() {
  std::string local;
  auto ups = getUniqueS(local);
  S* s = ups.get(); // expected-warning {{address of stack memory is returned later}}
  std::string_view sv = s->getData();
  return sv; // expected-note {{returned here}}
}

// FIXME: False negative. Move assignment of unique_ptr is not defaulted,
// so origins from `local` don't propagate to `ups`.
void owner_outlives_lifetimebound_source() {
  std::unique_ptr<S> ups;
  {
    std::string local;
    ups = getUniqueS(local);
  }
  (void)ups; // Should warn.
}

} // namespace track_origins_for_lifetimebound_record_type
