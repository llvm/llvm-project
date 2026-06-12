// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

int *identity(int *p __attribute__((lifetimebound))) { return p; }

struct PointerField {
  int *ptr;
};

struct IntField {
  int field;
};

void simple_case(void) {
  int *p;
  {
    int i;
    p = &i; // expected-warning {{local variable 'i' does not live long enough}}
  }         // expected-note {{destroyed here}}
  (void)*p; // expected-note {{later used here}}
}

void chained_assignment(void) {
  int *p, *q, *r;
  {
    int i;
    p = q = r = &i; // expected-warning {{local variable 'i' does not live long enough}}
  }                 // expected-note {{destroyed here}}
  (void)*p;         // expected-note {{later used here}}
}

void conditional_branch(int cond) {
  int safe;
  int *p = &safe;
  if (cond) {
    int i;
    p = &i; // expected-warning {{local variable 'i' does not live long enough}}
  }         // expected-note {{destroyed here}}
  (void)*p; // expected-note {{later used here}}
}

void loop_with_break(int cond) {
  int safe;
  int *p = &safe;
  for (int n = 0; n != 10; ++n) {
    if (cond) {
      int i;
      p = &i; // expected-warning {{local variable 'i' does not live long enough}}
      break;  // expected-note {{destroyed here}}
    }
  }
  (void)*p; // expected-note {{later used here}}
}

int *return_stack_address(void) {
  int i;
  int *p = &i; // expected-warning {{stack memory associated with local variable 'i' is returned}}
  return p;    // expected-note {{returned here}}
}

void lifetimebound_call(void) {
  int *p;
  {
    int i;
    p = identity(&i); // expected-warning {{local variable 'i' does not live long enough}} \
                      // expected-note {{expression aliases the storage of local variable 'i'}}
  }                   // expected-note {{destroyed here}}
  (void)*p;           // expected-note {{later used here}}
}

void struct_pointer_field(void) {
  int *p;
  {
    int i;
    struct PointerField holder;
    // FIXME: Track origins stored in struct pointer fields.
    holder.ptr = &i;
    p = holder.ptr;
  }
  (void)*p;
}

void struct_address_of_field(void) {
  int *p;
  {
    struct IntField holder;
    p = &holder.field; // expected-warning {{local variable 'holder' does not live long enough}}
  }                    // expected-note {{destroyed here}}
  (void)*p;            // expected-note {{later used here}}
}

void conditional_operator_lifetimebound(int cond) {
  int *p;
  {
    int a, b;
    p = identity(cond ? &a    // expected-warning {{local variable 'a' does not live long enough}}
                      : &b);  // expected-warning {{local variable 'b' does not live long enough}}
  }                           // expected-note 2 {{destroyed here}}
  (void)*p;                   // expected-note 2 {{later used here}}
}

union IntOrPtr {
  int i;
  int *p;
};

void union_member(void) {
  int *p;
  {
    union IntOrPtr u;
    p = &u.i; // expected-warning {{local variable 'u' does not live long enough}}
  }           // expected-note {{destroyed here}}
  (void)*p;   // expected-note {{later used here}}
}

struct AnonymousUnion {
  union {
    int i;
    float f;
  };
};

void anonymous_union_member(void) {
  int *p;
  {
    struct AnonymousUnion u;
    p = &u.i; // expected-warning {{local variable 'u' does not live long enough}}
  }           // expected-note {{destroyed here}}
  (void)*p;   // expected-note {{later used here}}
}

void function_address_regression(void) {
  extern void function_address_target(void);
  char *p = (char *)&function_address_target;
  (void)p;
}

int void_pointer_subscript_regression(void *bytes) {
  return &bytes[0] == &bytes[1];
}

typedef __attribute__((vector_size(16))) int v4i32;
v4i32 (*vector_factory)(int);

int vector_subscript_regression(void) {
  return (*vector_factory)(0)[0];
}

void va_arg_array_regression(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  int *p = __builtin_va_arg(ap, int[4]); // expected-warning {{second argument to 'va_arg' is of array type 'int[4]'}}
  (void)p;
}

void va_arg_function_regression(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  int (*p)(void) = __builtin_va_arg(ap, int(void)); // expected-error {{second argument to 'va_arg' is of non-POD type 'int (void)'}}
  (void)p;
}

// FIXME: We miss the origins of void* after dereference, so we miss to warn here.
void *void_pointer_dereference(void) {
  int value;
  void *bytes = &value;
  return &*bytes;
}

// FIXME: Atomics are not modeled yet.
int *atomic_pointer_declref(void) {
  int value;
  _Atomic(int *) p = &value;
  return p;
}
