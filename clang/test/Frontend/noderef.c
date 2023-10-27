// RUN: %clang_cc1 -Wno-unused-value -verify %s

#define NODEREF __attribute__((noderef))

struct S {
  int a;
  int b;
};

struct S2 {
  int a[2];
  int NODEREF a2[2];
  int *b;
  int NODEREF *b2;
  struct S *s;
  struct S NODEREF *s2;
};

int NODEREF *func(int NODEREF *arg) {  // expected-note{{arg declared here}}
  int y = *arg; // expected-warning{{dereferencing arg; was declared with a 'noderef' type}}
  return arg;
}

void func2(int x) {}

int test(void) {
  int NODEREF *p; // expected-note 34 {{p declared here}}
  int *p2;

  int x = *p;               // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  x = *((int NODEREF *)p2); // expected-warning{{dereferencing expression marked as 'noderef'}}

  int NODEREF **q;
  int *NODEREF *q2; // expected-note 4 {{q2 declared here}}

  // Indirection
  x = **q;  // expected-warning{{dereferencing expression marked as 'noderef'}}
  p2 = *q2; // expected-warning{{dereferencing q2; was declared with a 'noderef' type}}

  **q; // expected-warning{{dereferencing expression marked as 'noderef'}}

  p = *&*q;
  p = **&q;
  q = &**&q;
  p = &*p;
  p = *&p;
  p = &(*p);
  p = *(&p);
  x = **&p; // expected-warning{{dereferencing expression marked as 'noderef'}}

  *p = 2;   // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  *q = p;   // ok
  **q = 2;  // expected-warning{{dereferencing expression marked as 'noderef'}}
  *q2 = p2; // expected-warning{{dereferencing q2; was declared with a 'noderef' type}}

  p = p + 1;
  p = &*(p + 1);

  // Struct member access
  struct S NODEREF *s; // expected-note 3 {{s declared here}}
  x = s->a;   // expected-warning{{dereferencing s; was declared with a 'noderef' type}}
  x = (*s).b; // expected-warning{{dereferencing s; was declared with a 'noderef' type}}
  p = &s->a;
  p = &(*s).b;

  // Most things in sizeof() can't actually access memory
  x = sizeof(s->a);          // ok
  x = sizeof(*s);            // ok
  x = sizeof(s[0]);          // ok
  x = sizeof(s->a + (s->b)); // ok
  x = sizeof(int[++s->a]);   // expected-warning{{dereferencing s; was declared with a 'noderef' type}}

  // Struct member access should carry NoDeref type information through to an
  // enclosing AddrOf.
  p2 = &s->a;   // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  p2 = &(*s).a; // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  x = *&s->a;   // expected-warning{{dereferencing expression marked as 'noderef'}}

  // Nested struct access
  struct S2 NODEREF *s2_noderef;    // expected-note 5 {{s2_noderef declared here}}
  p = s2_noderef->a;  // ok since result is an array in a struct
  p = (*s2_noderef).a; // ok since result is an array in a struct
  p = s2_noderef->a2; // ok
  p = s2_noderef->b;  // expected-warning{{dereferencing s2_noderef; was declared with a 'noderef' type}}
  p = s2_noderef->b2; // expected-warning{{dereferencing s2_noderef; was declared with a 'noderef' type}}
  s = s2_noderef->s;  // expected-warning{{dereferencing s2_noderef; was declared with a 'noderef' type}}
  s = s2_noderef->s2; // expected-warning{{dereferencing s2_noderef; was declared with a 'noderef' type}}
  p = s2_noderef->a + 1;

  struct S2 *s2;
  p = s2->a;
  p = s2->a2;
  p = s2->b;
  p = s2->b2;
  s = s2->s;
  s = s2->s2;
  &(*(*s2).s2).b;

  // Subscript access
  x = p[1];    // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  x = q[0][0]; // expected-warning{{dereferencing expression marked as 'noderef'}}
  p2 = q2[0];  // expected-warning{{dereferencing q2; was declared with a 'noderef' type}}
  p = q[*p];   // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  x = p[*p];   // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
               // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}

  int NODEREF arr[10];    // expected-note 1 {{arr declared here}}
  x = arr[1]; // expected-warning{{dereferencing arr; was declared with a 'noderef' type}}

  int NODEREF *(arr2[10]);
  int NODEREF *elem = *arr2;

  int NODEREF(*arr3)[10];
  elem = *arr3;

  // Combinations between indirection, subscript, and member access
  struct S2 NODEREF *s2_arr[10];
  struct S2 NODEREF *s2_arr2[10][10];

  p = s2_arr[1]->a;
  p = s2_arr[1]->b; // expected-warning{{dereferencing expression marked as 'noderef'}}
  int *NODEREF *bptr = &s2_arr[1]->b;

  x = s2->s2->a;        // expected-warning{{dereferencing expression marked as 'noderef'}}
  x = s2_noderef->a[1]; // expected-warning{{dereferencing s2_noderef; was declared with a 'noderef' type}}
  p = &s2_noderef->a[1];

  // Casting to dereferenceable pointer
  p2 = p;             // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  p2 = *q;            // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  p2 = q[0];          // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  s2 = s2_arr[1];     // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  s2 = s2_arr2[1][1]; // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  p2 = p, p2 = *q;    // expected-warning 2 {{casting to dereferenceable pointer removes 'noderef' attribute}}

  // typedefs
  typedef int NODEREF *ptr_t;
  ptr_t ptr; // expected-note 2 {{ptr declared here}}
  ptr_t *ptr2;
  *ptr; // expected-warning{{dereferencing ptr; was declared with a 'noderef' type}}
  *ptr2;
  **ptr2; // expected-warning{{dereferencing expression marked as 'noderef'}}

  typedef struct S2 NODEREF *s2_ptr_t;
  s2_ptr_t s2_ptr; // expected-note 4 {{s2_ptr declared here}}
  s2_ptr->a;       // ok since result is an array in a struct
  s2_ptr->a2;      // ok
  s2_ptr->b;       // expected-warning{{dereferencing s2_ptr; was declared with a 'noderef' type}}
  s2_ptr->b2;      // expected-warning{{dereferencing s2_ptr; was declared with a 'noderef' type}}
  s2_ptr->s;       // expected-warning{{dereferencing s2_ptr; was declared with a 'noderef' type}}
  s2_ptr->s2;      // expected-warning{{dereferencing s2_ptr; was declared with a 'noderef' type}}
  s2_ptr->a + 1;

  typedef int(int_t);
  typedef int_t NODEREF *(noderef_int_t);
  typedef noderef_int_t *noderef_int_nested_t;
  noderef_int_nested_t noderef_int_nested_ptr;
  *noderef_int_nested_ptr;
  **noderef_int_nested_ptr; // expected-warning{{dereferencing expression marked as 'noderef'}}

  typedef int_t *(NODEREF noderef_int2_t);
  typedef noderef_int2_t *noderef_int2_nested_t;
  noderef_int2_nested_t noderef_int2_nested_ptr; // expected-note{{noderef_int2_nested_ptr declared here}}
  *noderef_int2_nested_ptr;                      // expected-warning{{dereferencing noderef_int2_nested_ptr; was declared with a 'noderef' type}}

  typedef int_t *(noderef_int3_t);
  typedef noderef_int3_t(NODEREF(*(noderef_int3_nested_t)));
  noderef_int3_nested_t noderef_int3_nested_ptr; // expected-note{{noderef_int3_nested_ptr declared here}}
  *noderef_int3_nested_ptr;                      // expected-warning{{dereferencing noderef_int3_nested_ptr; was declared with a 'noderef' type}}

  // Parentheses
  (((*((p))))); // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  (*(*(&(p)))); // expected-warning{{dereferencing expression marked as 'noderef'}}

  (p[1]);      // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  (q[0]);      // ok
  (q[0][0]);   // expected-warning{{dereferencing expression marked as 'noderef'}}
  (q2[0]);     // expected-warning{{dereferencing q2; was declared with a 'noderef' type}}
  (q[(*(p))]); // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  (p[(*(p))]); // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
               // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}

  (*(ptr)); // expected-warning{{dereferencing ptr; was declared with a 'noderef' type}}
  (*(ptr2));
  (*(*(ptr2))); // expected-warning{{dereferencing expression marked as 'noderef'}}

  // Functions
  x = *(func(p)); // expected-warning{{dereferencing expression marked as 'noderef'}}

  // Casting is ok
  q = (int NODEREF **)&p;
  q = (int NODEREF **)&p2;
  q = &p;
  q = &p2;
  x = s2->s2->a; // expected-warning{{dereferencing expression marked as 'noderef'}}

  // Other expressions
  func2(*p);         // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  func2(*p + 1);     // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  func2(!*p);        // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  func2((x = *p));   // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  func2((char)(*p)); // expected-warning{{dereferencing p; was declared with a 'noderef' type}}

  // Other statements
  if (*p) {}          // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  else if (*p) {}     // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  switch (*p){}       // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  for (*p; *p; *p){}  // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
                      // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}
                      // expected-warning@-2{{dereferencing p; was declared with a 'noderef' type}}
  for (*p; *p;){}     // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
                      // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}
  for (*p;; *p){}     // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
                      // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}
  for (; *p; *p){}    // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
                      // expected-warning@-1{{dereferencing p; was declared with a 'noderef' type}}
  for (*p;;){}        // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  for (;*p;){}        // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  for (;;*p){}        // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  while (*p){}        // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  do {} while (*p);   // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
  return *p;          // expected-warning{{dereferencing p; was declared with a 'noderef' type}}
}

// FIXME: Currently, [[]] syntax does not work for the `noderef` atribute.
// For the time being, test that we consistently diagnose the attribute as
// ignored.
// For details see https://github.com/llvm/llvm-project/issues/55790
void test_standard_syntax() {
  [[clang::noderef]] int i; // expected-warning {{'noderef' attribute ignored}}

  [[clang::noderef]] int *p1; // expected-warning {{'noderef' attribute ignored}}
  *p1;

  int *p2 [[clang::noderef]]; // expected-warning {{'noderef' attribute ignored}}
  *p2;

  int * [[clang::noderef]] p3; // expected-warning {{'noderef' attribute ignored}}
  *p3;

  typedef int* IntPtr;
  [[clang::noderef]] IntPtr p4; // expected-warning {{'noderef' attribute ignored}}
  *p4;
}
