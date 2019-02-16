// RUN: %clang_analyze_cc1 -x c++ -analyzer-checker=core -analyzer-output=text -verify %s

int initializer1(int &p, int x) {
  if (x) { // expected-note{{Taking false branch}}
    p = 1;
    return 0;
  } else {
    return 1; // expected-note {{Returning without writing to 'p'}}
  }
}

int param_not_initialized_by_func() {
  int p;                        // expected-note {{'p' declared without an initial value}}
  int out = initializer1(p, 0); // expected-note{{Calling 'initializer1'}}
                                // expected-note@-1{{Returning from 'initializer1'}}
  return p;                     // expected-note{{Undefined or garbage value returned to caller}}
                                // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

struct S {
  int initialize(int *p, int param) {
    if (param) { //expected-note{{Taking false branch}}
      *p = 1;
      return 1;
    }
    return 0; // expected-note{{Returning without writing to '*p'}}
  }
};

int use(S *s) {
  int p;                //expected-note{{'p' declared without an initial value}}
  s->initialize(&p, 0); //expected-note{{Calling 'S::initialize'}}
                        //expected-note@-1{{Returning from 'S::initialize'}}
  return p;             // expected-warning{{Undefined or garbage value returned to caller}}
                        // expected-note@-1{{Undefined or garbage value returned to caller}}
}

int initializer2(const int &p) {
  return 0;
}

int no_msg_const_ref() {
  int p; //expected-note{{'p' declared without an initial value}}
  initializer2(p);
  return p; // expected-warning{{Undefined or garbage value returned to caller}}
            // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void nested() {}
void init_in_nested_func(int **x) {
  *x = 0; // expected-note{{Null pointer value stored to 'y'}}
  nested();
} // no-note

int call_init_nested() {
  int x = 0;
  int *y = &x;
  init_in_nested_func(&y); // expected-note{{Calling 'init_in_nested_func'}}
                           // expected-note@-1{{Returning from 'init_in_nested_func'}}
  return *y;               //expected-warning{{Dereference of null pointer (loaded from variable 'y')}}
                           //expected-note@-1{{Dereference of null pointer (loaded from variable 'y')}}
}

struct A {
  int x;
  int y;
};

void partial_init_by_reference(A &a) {
  a.x = 0;
} // expected-note {{Returning without writing to 'a.y'}}

int use_partial_init_by_reference() {
  A a;
  partial_init_by_reference(a); // expected-note{{Calling 'partial_init_by_reference'}}
                                // expected-note@-1{{Returning from 'partial_init_by_reference'}}
  return a.y;                   // expected-warning{{Undefined or garbage value returned to caller}}
                                // expected-note@-1{{Undefined or garbage value returned to caller}}
}

struct B : A {
};

void partially_init_inherited_struct(B *b) {
  b->x = 0;
} // expected-note{{Returning without writing to 'b->y'}}

int use_partially_init_inherited_struct() {
  B b;
  partially_init_inherited_struct(&b); // expected-note{{Calling 'partially_init_inherited_struct'}}
                                       // expected-note@-1{{Returning from 'partially_init_inherited_struct'}}
  return b.y;                          // expected-warning{{Undefined or garbage value returned to caller}}
                                       // expected-note@-1{{Undefined or garbage value returned to caller}}
}

struct C {
  int x;
  int y;
  C(int pX, int pY) : x(pX) {} // expected-note{{Returning without writing to 'this->y'}}

  C(int pX, int pY, bool Flag) {
    x = pX;
    if (Flag) // expected-note{{Assuming 'Flag' is not equal to 0}}
              // expected-note@-1{{Taking true branch}}
      return; // expected-note{{Returning without writing to 'this->y'}}
    y = pY;
  }
};

int use_constructor() {
  C c(0, 0); // expected-note{{Calling constructor for 'C'}}
             // expected-note@-1{{Returning from constructor for 'C'}}
  return c.y; // expected-note{{Undefined or garbage value returned to caller}}
              // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

int coin();

int use_other_constructor() {
  C c(0, 0, coin()); // expected-note{{Calling constructor for 'C'}}
                     // expected-note@-1{{Returning from constructor for 'C'}}
  return c.y; // expected-note{{Undefined or garbage value returned to caller}}
              // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

struct D {
  void initialize(int *);
};

void D::initialize(int *p) {

} // expected-note{{Returning without writing to '*p'}}

int use_d_initializer(D* d) {
  int p; // expected-note {{'p' declared without an initial value}}
  d->initialize(&p); // expected-note{{Calling 'D::initialize'}}
                     // expected-note@-1{{Returning from 'D::initialize'}}
  return p;                     // expected-note{{Undefined or garbage value returned to caller}}
                                // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

struct S2 {
  int x;
};

int pointerreference(S2* &s) {
  if (coin()) // expected-note{{Assuming the condition is true}}
              // expected-note@-1{{Taking true branch}}
    return 1; // expected-note{{Returning without writing to 's->x'}}

  s->x = 0;
  return 0;
}

int usepointerreference() {
  S2 s;
  S2* p = &s;
  pointerreference(p); //expected-note{{Calling 'pointerreference'}}
                         //expected-note@-1{{Returning from 'pointerreference'}}
  return s.x; // expected-warning{{Undefined or garbage value returned to caller}}
              // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void *has_no_argument_and_returns_null(void) {
  return 0;
}

void rdar40335545() {
    int local; // expected-note{{}}
    void (*takes_int_ptr_argument)(int *) = (void (*)(int*))has_no_argument_and_returns_null;

    takes_int_ptr_argument(&local); // no-crash

    int useLocal = local; //expected-warning{{}}
                          //expected-note@-1{{}}
    (void)useLocal;
}
