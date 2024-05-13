// RUN: %clang_analyze_cc1 -w -Wno-implicit-function-declaration -analyzer-checker=core -analyzer-output=text\
// RUN:     -verify %s

typedef __typeof(sizeof(int)) size_t;
void *memset(void *__s, int __c, size_t __n);

int initializer1(int *p, int x) {
  if (x) { // expected-note{{'x' is 0}}
           // expected-note@-1{{Taking false branch}}
    *p = 1;
    return 0;
  } else {
    return 1; // expected-note {{Returning without writing to '*p'}}
  }
}

int param_not_initialized_by_func(void) {
  int p; // expected-note {{'p' declared without an initial value}}
  int out = initializer1(&p, 0); // expected-note{{Calling 'initializer1'}}
                                 // expected-note@-1{{Returning from 'initializer1'}}
  return p; // expected-note{{Undefined or garbage value returned to caller}}
            // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

int param_initialized_properly(void) {
  int p;
  int out = initializer1(&p, 1);
  return p; //no-warning
}

static int global;

int initializer2(int **p, int x) {
  if (x) { // expected-note{{'x' is 0}}
           // expected-note@-1{{Taking false branch}}
    *p = &global;
    return 0;
  } else {
    return 1; // expected-note {{Returning without writing to '*p'}}
  }
}

int param_not_written_into_by_func(void) {
  int *p = 0;                    // expected-note{{'p' initialized to a null pointer value}}
  int out = initializer2(&p, 0); // expected-note{{Calling 'initializer2'}}
                                 // expected-note@-1{{Returning from 'initializer2'}}
  return *p;                     // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
                                 // expected-note@-1{{Dereference of null pointer (loaded from variable 'p')}}
}

void initializer3(int *p, int param) {
  if (param) // expected-note{{'param' is 0}}
             // expected-note@-1{{Taking false branch}}
    *p = 0;
} // expected-note{{Returning without writing to '*p'}}

int param_written_into_by_void_func(void) {
  int p;               // expected-note{{'p' declared without an initial value}}
  initializer3(&p, 0); // expected-note{{Calling 'initializer3'}}
                       // expected-note@-1{{Returning from 'initializer3'}}
  return p;            // expected-warning{{Undefined or garbage value returned to caller}}
                       // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void initializer4(int *p, int param) {
  if (param) // expected-note{{'param' is 0}}
             // expected-note@-1{{Taking false branch}}
    *p = 0;
} // expected-note{{Returning without writing to '*p'}}

void initializer5(int *p, int param) {
  if (!param) // expected-note{{'param' is 1}}
              // expected-note@-1{{Taking false branch}}
    *p = 0;
} // expected-note{{Returning without writing to '*p'}}

int multi_init_tries_func(void) {
  int p;               // expected-note{{'p' declared without an initial value}}
  initializer4(&p, 0); // expected-note{{Calling 'initializer4'}}
                       // expected-note@-1{{Returning from 'initializer4'}}
  initializer5(&p, 1); // expected-note{{Calling 'initializer5'}}
                       // expected-note@-1{{Returning from 'initializer5'}}
  return p;            // expected-warning{{Undefined or garbage value returned to caller}}
                       // expected-note@-1{{Undefined or garbage value returned to caller}}
}

int initializer6(const int *p) {
  return 0;
}

int no_msg_on_const(void) {
  int p; // expected-note{{'p' declared without an initial value}}
  initializer6(&p);
  return p; // expected-warning{{Undefined or garbage value returned to caller}}
            // expected-note@-1{{Undefined or garbage value returned to caller}}
}

typedef struct {
  int x;
} S;

int initializer7(S *s, int param) {
  if (param) { // expected-note{{'param' is 0}}
               // expected-note@-1{{Taking false branch}}
    s->x = 0;
    return 0;
  }
  return 1; // expected-note{{Returning without writing to 's->x'}}
}

int initialize_struct_field(void) {
  S local;
  initializer7(&local, 0); // expected-note{{Calling 'initializer7'}}
                           // expected-note@-1{{Returning from 'initializer7'}}
  return local.x;          // expected-warning{{Undefined or garbage value returned to caller}}
                           // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void nullwriter(int **p) {
  *p = 0; // expected-note{{Null pointer value stored to 'p'}}
} // no extra note

int usage(void) {
  int x = 0;
  int *p = &x;
  nullwriter(&p); // expected-note{{Calling 'nullwriter'}}
                  // expected-note@-1{{Returning from 'nullwriter'}}
  return *p;      // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
                  // expected-note@-1{{Dereference of null pointer (loaded from variable 'p')}}
}

typedef struct {
  int x;
  int y;
} A;

void partial_initializer(A *a) {
  a->x = 0;
} // expected-note{{Returning without writing to 'a->y'}}

int use_partial_initializer(void) {
  A a;
  partial_initializer(&a); // expected-note{{Calling 'partial_initializer'}}
                           // expected-note@-1{{Returning from 'partial_initializer'}}
  return a.y;              // expected-warning{{Undefined or garbage value returned to caller}}
                           // expected-note@-1{{Undefined or garbage value returned to caller}}
}

typedef struct {
  int x;
  int y;
} B;

typedef struct {
  B b;
} C;

void partial_nested_initializer(C *c) {
  c->b.x = 0;
} // expected-note{{Returning without writing to 'c->b.y'}}

int use_partial_nested_initializer(void) {
  B localB;
  C localC;
  localC.b = localB;
  partial_nested_initializer(&localC); // expected-note{{Calling 'partial_nested_initializer'}}
                                       // expected-note@-1{{Returning from 'partial_nested_initializer'}}
  return localC.b.y;                   // expected-warning{{Undefined or garbage value returned to caller}}
                                       // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void test_subregion_assignment(C* c) {
  B b;
  c->b = b;
}

int use_subregion_assignment(void) {
  C c;
  test_subregion_assignment(&c); // expected-note{{Calling 'test_subregion_assignment'}}
                                 // expected-note@-1{{Returning from 'test_subregion_assignment'}}
  return c.b.x; // expected-warning{{Undefined or garbage value returned to caller}}
                // expected-note@-1{{Undefined or garbage value returned to caller}}
}

int confusing_signature(int *);
int confusing_signature(int *p) {
  return 0; // expected-note{{Returning without writing to '*p'}}
}

int use_confusing_signature(void) {
  int a; // expected-note {{'a' declared without an initial value}}
  confusing_signature(&a); // expected-note{{Calling 'confusing_signature'}}
                           // expected-note@-1{{Returning from 'confusing_signature'}}
  return a; // expected-note{{Undefined or garbage value returned to caller}}
            // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

int coin(void);

int multiindirection(int **p) {
  if (coin()) // expected-note{{Assuming the condition is true}}
              // expected-note@-1{{Taking true branch}}
    return 1; // expected-note{{Returning without writing to '**p'}}
  *(*p) = 0;
  return 0;
}

int usemultiindirection(void) {
  int a; // expected-note {{'a' declared without an initial value}}
  int *b = &a;
  multiindirection(&b); // expected-note{{Calling 'multiindirection'}}
                        // expected-note@-1{{Returning from 'multiindirection'}}
  return a; // expected-note{{Undefined or garbage value returned to caller}}
            // expected-warning@-1{{Undefined or garbage value returned to caller}}
}

int indirectingstruct(S** s) {
  if (coin()) // expected-note{{Assuming the condition is true}}
              // expected-note@-1{{Taking true branch}}
    return 1; // expected-note{{Returning without writing to '(*s)->x'}}

  (*s)->x = 0;
  return 0;
}

int useindirectingstruct(void) {
  S s;
  S* p = &s;
  indirectingstruct(&p); //expected-note{{Calling 'indirectingstruct'}}
                         //expected-note@-1{{Returning from 'indirectingstruct'}}
  return s.x; // expected-warning{{Undefined or garbage value returned to caller}}
              // expected-note@-1{{Undefined or garbage value returned to caller}}
}

typedef struct {
  int *x;
} D;

void initializeMaybeInStruct(D* pD) {
  if (coin()) // expected-note{{Assuming the condition is false}}
              // expected-note@-1{{Taking false branch}}
    *pD->x = 120;
} // expected-note{{Returning without writing to 'pD->x'}}

int useInitializeMaybeInStruct(void) {
  int z; // expected-note{{'z' declared without an initial value}}
  D d;
  d.x = &z;
  initializeMaybeInStruct(&d); // expected-note{{Calling 'initializeMaybeInStruct'}}
                               // expected-note@-1{{Returning from 'initializeMaybeInStruct'}}
  return z; // expected-warning{{Undefined or garbage value returned to caller}}
            // expected-note@-1{{Undefined or garbage value returned to caller}}
}

void test_implicit_function_decl(int *x) {
  if (x) {} // expected-note{{Assuming 'x' is null}}
            // expected-note@-1{{Taking false branch}}
  implicit_function(x);
  *x = 4; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
          // expected-note@-1{{Dereference of null pointer (loaded from variable 'x')}}
}
int implicit_function(int *y) {}
