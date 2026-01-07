// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -Wexperimental-lifetime-safety -Wno-dangling -verify %s

struct View;

struct [[gsl::Owner]] MyObj {
  int id;
  ~MyObj() {}  // Non-trivial destructor
  MyObj operator+(MyObj);
  
  View getView() const [[clang::lifetimebound]];
};

struct [[gsl::Pointer()]] View {
  View(const MyObj&); // Borrows from MyObj
  View();
  void use() const;
};

class TriviallyDestructedClass {
  View a, b;
};

//===----------------------------------------------------------------------===//
// Basic Definite Use-After-Free (-W...permissive)
// These are cases where the pointer is guaranteed to be dangling at the use site.
//===----------------------------------------------------------------------===//

void definite_simple_case() {
  MyObj* p;
  {
    MyObj s;
    p = &s;     // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
}

void definite_simple_case_gsl() {
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

void definite_pointer_chain() {
  MyObj* p;
  MyObj* q;
  {
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
    q = p;
  }             // expected-note {{destroyed here}}
  (void)*q;     // expected-note {{later used here}}
}

void definite_propagation_gsl() {
  View v1, v2;
  {
    MyObj s;
    v1 = s;     // expected-warning {{object whose reference is captured does not live long enough}}
    v2 = v1;
  }             // expected-note {{destroyed here}}
  v2.use();     // expected-note {{later used here}}
}

void definite_multiple_uses_one_warning() {
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

void definite_multiple_pointers() {
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

void definite_single_pointer_multiple_loans(bool cond) {
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

void definite_single_pointer_multiple_loans_gsl(bool cond) {
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

void definite_if_branch(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  if (cond) {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;     // expected-note {{later used here}}
}

void potential_if_branch(bool cond) {
  MyObj safe;
  MyObj* p = &safe;
  if (cond) {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured may not live long enough}}
  }             // expected-note {{destroyed here}}
  if (!cond)
    (void)*p;   // expected-note {{later used here}}
  else
    p = &safe;
}

void definite_if_branch_gsl(bool cond) {
  MyObj safe;
  View v = safe;
  if (cond) {
    MyObj temp;
    v = temp;   // expected-warning {{object whose reference is captured does not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note {{later used here}}
}

void definite_potential_together(bool cond) {
  MyObj safe;
  MyObj* p_maybe = &safe;
  MyObj* p_definite = nullptr;

  {
    MyObj s;
    if (cond)
      p_definite = &s;  // expected-warning {{does not live long enough}}
    if (cond)
      p_maybe = &s;     // expected-warning {{may not live long enough}}         
  }                     // expected-note 2 {{destroyed here}}
  (void)*p_definite;    // expected-note {{later used here}}
  if (!cond)
    (void)*p_maybe;     // expected-note {{later used here}}
}

void definite_overrides_potential(bool cond) {
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

  // The use of 'p' is a definite error because it was never rescued.
  (void)*q;
  (void)*p;       // expected-note {{later used here}}
  (void)*q;
}

void potential_due_to_conditional_killing(bool cond) {
  MyObj safe;
  MyObj* q;
  {
    MyObj s;
    q = &s;       // expected-warning {{may not live long enough}}
  }               // expected-note {{destroyed here}}
  if (cond) {
    // 'q' is conditionally "rescued". 'p' is not.
    q = &safe;
  }
  (void)*q;       // expected-note {{later used here}}
}

void potential_for_loop_use_after_loop_body(MyObj safe) {
  MyObj* p = &safe;
  for (int i = 0; i < 1; ++i) {
    MyObj s;
    p = &s;     // expected-warning {{may not live long enough}}
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

void potential_for_loop_gsl() {
  MyObj safe;
  View v = safe;
  for (int i = 0; i < 1; ++i) {
    MyObj s;
    v = s;      // expected-warning {{object whose reference is captured may not live long enough}}
  }             // expected-note {{destroyed here}}
  v.use();      // expected-note {{later used here}}
}

void potential_for_loop_use_before_loop_body(MyObj safe) {
  MyObj* p = &safe;
  // Prefer the earlier use for diagnsotics.
  for (int i = 0; i < 1; ++i) {
    (void)*p;   // expected-note {{later used here}}
    MyObj s;
    p = &s;     // expected-warning {{does not live long enough}}
  }             // expected-note {{destroyed here}}
  (void)*p;
}

void definite_loop_with_break(bool cond) {
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

void definite_loop_with_break_gsl(bool cond) {
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

void potential_multiple_expiry_of_same_loan(bool cond) {
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

  // TODO: This can be argued to be a "maybe" warning. This is because
  // we only check for confidence of liveness and not the confidence of
  // the loan contained in an origin. To deal with this, we can introduce
  // a confidence in loan propagation analysis as well like liveness.
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

void potential_switch(int mode) {
  MyObj safe;
  MyObj* p = &safe;
  switch (mode) {
  case 1: {
    MyObj temp;
    p = &temp;  // expected-warning {{object whose reference is captured may not live long enough}}
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

void definite_switch(int mode) {
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

void definite_switch_gsl(int mode) {
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
    p = &x;     // expected-warning {{may not live long enough}}

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
// Basic Definite Use-After-Return (Return-Stack-Address) (-W...permissive)
// These are cases where the pointer is guaranteed to be dangling at the use site.
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
    p = local_obj;  // expected-warning {{address of stack memory is returned later}}
    if (c) {
      return p;      // expected-note {{returned here}}
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

// FIXME: No warning for non gsl::Pointer types. Origin tracking is only supported for pointer types.
struct LifetimeBoundCtor {
  LifetimeBoundCtor();
  LifetimeBoundCtor(const MyObj& obj [[clang::lifetimebound]]);
};

void lifetimebound_ctor() {
  LifetimeBoundCtor v;
  {
    MyObj obj;
    v = obj;
  }
  (void)v;
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
  lifetimebound_return_by_value_param_template(MyObj{}); // expected-note {{in instantiation of}}
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
    p = cond ? &temp  // expected-warning {{object whose reference is captured may not live long enough}}
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
      v = o; // expected-warning {{object whose reference is captured may not live long enough}}
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
