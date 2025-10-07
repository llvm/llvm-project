// RUN: %clang_cc1 -fsyntax-only -fexperimental-lifetime-safety -Wexperimental-lifetime-safety -verify %s

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


//===----------------------------------------------------------------------===//
// Lifetimebound Attribute Tests
//===----------------------------------------------------------------------===//

View Identity(View v [[clang::lifetimebound]]);
View Choose(bool cond, View a [[clang::lifetimebound]], View b [[clang::lifetimebound]]);
MyObj* GetPointer(const MyObj& obj [[clang::lifetimebound]]);

struct [[gsl::Pointer()]] LifetimeBoundView {
  LifetimeBoundView();
  LifetimeBoundView(const MyObj& obj [[clang::lifetimebound]]);
  LifetimeBoundView pass() [[clang::lifetimebound]] { return *this; }
  operator View() const [[clang::lifetimebound]];
};

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

void lifetimebound_member_function() {
  LifetimeBoundView lbv, lbv2;
  {
    MyObj obj;
    lbv = obj;        // expected-warning {{object whose reference is captured does not live long enough}}
    lbv2 = lbv.pass();
  }                   // expected-note {{destroyed here}}
  View v = lbv2;      // expected-note {{later used here}}
  v.use();
}

void lifetimebound_conversion_operator() {
  View v;
  {
    MyObj obj;
    LifetimeBoundView lbv = obj; // expected-warning {{object whose reference is captured does not live long enough}}
    v = lbv;                     // Conversion operator is lifetimebound
  }                              // expected-note {{destroyed here}}
  v.use();                       // expected-note {{later used here}}
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

// FIXME: Creating reference from lifetimebound call doesn't propagate loans.
const MyObj& GetObject(View v [[clang::lifetimebound]]);
void lifetimebound_return_reference() {
  View v;
  const MyObj* ptr;
  {
    MyObj obj;
    View temp_v = obj;
    const MyObj& ref = GetObject(temp_v);
    ptr = &ref;
  }
  (void)*ptr;
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
