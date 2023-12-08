// RUN: clang-tidy -checks=-*,modernize-redundant-void-arg %s -- -Wno-strict-prototypes -x c | count 0

#define NULL 0

extern int i;

int foo2() {
  return 0;
}

int j = 1;

int foo(void) {
  return 0;
}

typedef unsigned int my_uint;

typedef void my_void;

// A function taking void and returning a pointer to function taking void
// and returning int.
int (*returns_fn_void_int(void))(void);

typedef int (*returns_fn_void_int_t(void))(void);

int (*returns_fn_void_int(void))(void) {
  return NULL;
}

// A function taking void and returning a pointer to a function taking void
// and returning a pointer to a function taking void and returning void.
void (*(*returns_fn_returns_fn_void_void(void))(void))(void);

typedef void (*(*returns_fn_returns_fn_void_void_t(void))(void))(void);

void (*(*returns_fn_returns_fn_void_void(void))(void))(void) {
  return NULL;
}

void bar(void) {
  int i;
  int *pi = NULL;
  void *pv = (void *) pi;
  float f;
  float *fi;
  double d;
  double *pd;
}

void (*f1)(void);
void (*f2)(void) = NULL;
void (*f3)(void) = bar;
void (*fa)();
void (*fb)() = NULL;
void (*fc)() = bar;

typedef void (function_ptr)(void);
