// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -Wno-varargs -Wno-non-pod-varargs -verify -fexperimental-lifetime-safety-c %s
// RUN: %clang_cc1 -fsyntax-only -Werror=lifetime-safety -Wno-dangling -Wno-varargs -Wno-non-pod-varargs %s

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
  }         // expected-note {{local variable 'i' is destroyed here}}
  (void)*p; // expected-note {{later used here}}
}

void chained_assignment(void) {
  int *p, *q, *r;
  {
    int i;
    p = q = r = &i; // expected-warning {{local variable 'i' does not live long enough}}
  }                 // expected-note {{local variable 'i' is destroyed here}}
  (void)*p;         // expected-note {{later used here}}
}

void conditional_branch(int cond) {
  int safe;
  int *p = &safe;
  if (cond) {
    int i;
    p = &i; // expected-warning {{local variable 'i' does not live long enough}}
  }         // expected-note {{local variable 'i' is destroyed here}}
  (void)*p; // expected-note {{later used here}}
}

void loop_with_break(int cond) {
  int safe;
  int *p = &safe;
  for (int n = 0; n != 10; ++n) {
    if (cond) {
      int i;
      p = &i; // expected-warning {{local variable 'i' does not live long enough}}
      break;  // expected-note {{local variable 'i' is destroyed here}}
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
                      // expected-note {{result of call to 'identity' aliases the storage of local variable 'i'}}
  }                   // expected-note {{local variable 'i' is destroyed here}}
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
  }                    // expected-note {{local variable 'holder' is destroyed here}}
  (void)*p;            // expected-note {{later used here}}
}

void conditional_operator_lifetimebound(int cond) {
  int *p;
  {
    int a, b;
    p = identity(cond ? &a    // expected-warning {{local variable 'a' does not live long enough}} \
                              // expected-note {{result of call to 'identity' aliases the storage of local variable 'a'}} \
                              // expected-note {{result of call to 'identity' aliases the storage of local variable 'b'}}
                      : &b);  // expected-warning {{local variable 'b' does not live long enough}}
  }                           // expected-note {{local variable 'a' is destroyed here}} \
                              // expected-note {{local variable 'b' is destroyed here}}
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
  }           // expected-note {{local variable 'u' is destroyed here}}
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
  }           // expected-note {{local variable 'u' is destroyed here}}
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
  int *p = __builtin_va_arg(ap, int[4]);
  (void)p;
}

void take(int* q);
void va_arg_array_paren_regression(int n, ...) {
  __builtin_va_list ap;
  take((__builtin_va_arg(ap, int[4])));
}

void va_arg_function_regression(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  int (*p)(void) = __builtin_va_arg(ap, int(void));
  (void)p;
}

// FIXME: We miss the origins of void* after dereference, so we miss to warn here.
void *void_pointer_dereference(void) {
  int value;
  void *bytes = &value;
  return &*bytes;
}

// `_Atomic(T)` is transparent for lifetime purposes; a stack address laundered
// through an atomic is caught.
int *atomic_pointer_declref(void) {
  int value;
  _Atomic(int *) p = &value; // expected-warning {{stack memory associated with local variable 'value' is returned}}
  return p;                  // expected-note {{returned here}}
}

int *atomic_pointer_static(void) {
  static int value;
  _Atomic(int *) p = &value;
  return p; // no-warning
}

int **atomic_pointer_multilevel(void) {
  int *inner;
  _Atomic(int **) p = &inner; // expected-warning {{stack memory associated with local variable 'inner' is returned}}
  return p;                   // expected-note {{returned here}}
}

// In C, a pointer compound assignment is a prvalue; its result still carries
// the LHS pointer's loans.
void compound_assign_prvalue(void) {
  int *p;
  {
    int local[10];
    int *q = local; // expected-warning {{local variable 'local' does not live long enough}}
    p = (q += 1);
  }               // expected-note {{destroyed here}}
  (void)*p;       // expected-note {{later used here}}
}

void preincrement_prvalue(void) {
  int *p;
  {
    int local[10];
    int *q = local; // expected-warning {{local variable 'local' does not live long enough}}
    p = ++q;
  }               // expected-note {{destroyed here}}
  (void)*p;       // expected-note {{later used here}}
}
