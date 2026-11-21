// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 -Wthread-safety -Wthread-safety-beta %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 -Wthread-safety -Wthread-safety-beta %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wthread-safety -Wthread-safety-beta %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wthread-safety %s

// Tests for local variables whose value may change behind the analysis'
// back: call-argument escapes and constructor arguments.

struct __attribute__((capability("mutex"))) Mutex {
  int x;
};

struct Mutex mu;
int data __attribute__((guarded_by(mu)));

void unlock(struct Mutex *m) __attribute__((release_capability(m)));
int trylock(struct Mutex *m) __attribute__((try_acquire_capability(1, m)));
int cond(void);

int trylock_zero() __attribute__((try_acquire_capability(0, mu)));

//===----------------------------------------------------------------------===//
// Call-argument escapes and constructor arguments.
//===----------------------------------------------------------------------===//

int *id_ptr_alias(int *p) { return p; }
void take_uintptr(__UINTPTR_TYPE__);
void take_const_ptr_ref(int *const &);
void bind(int *);
int *global_escaped_r;
void keep(int *p) { global_escaped_r = p; }
void poke() { *global_escaped_r = 0; }

void test_lvalue_and_call_escapes() {
  int r0;
  int *p0 = &(r0 = trylock_zero());
  *p0 = 0;
  if (r0 == 0) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  auto set = [](int &x, int val) { x = val; };
  int r1 = trylock_zero();
  set(r1, 0);
  if (r1 == 0) {
    data = 2; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  int r2 = 1;
  int *p2 = id_ptr_alias(&r2);
  r2 = trylock_zero();
  *p2 = 0;
  if (!r2) {
    data = 3; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  int r3 = trylock_zero();
  take_uintptr((__UINTPTR_TYPE__)&r3);
  if (!r3) {
    data = 4; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  int r4 = trylock_zero();
  take_const_ptr_ref(&r4);
  if (!r4) {
    data = 5; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  int r5 = trylock_zero();
  bind(&r5);
  if (!r5) {
    data = 6; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }

  int r6 = 1;
  keep(&r6);
  r6 = trylock_zero();
  poke();
  if (r6 == 0) {
    data = 7; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }
}

// Constructor mutating a local pointer by reference invalidates
// its definition in LocalVariableMap.
Mutex mu_ctor_other;
struct RedirectMutexPtr {
  RedirectMutexPtr(Mutex *&p) { p = &mu_ctor_other; }
};
void test_ctor_mutates_local_ptr() {
  Mutex *p = &mu;
  RedirectMutexPtr redir(p);
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}

// A conditional or comma lvalue passed by mutable reference or whose address
// is passed invalidates the definitions of all variables it may designate.
void redirect_ref(Mutex *&p);
void redirect_ptr(Mutex **pp);
void set_int(int &x);
void set_int_ptr(int *x);

void test_conditional_lvalue_ref_arg(bool c) {
  Mutex *p = &mu, *q = &mu;
  redirect_ref(c ? p : q);
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}

void test_comma_lvalue_ref_arg() {
  Mutex *p = &mu;
  redirect_ref((cond(), p));
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}

void test_conditional_lvalue_addr_arg(bool c) {
  Mutex *p = &mu, *q = &mu;
  redirect_ptr(&(c ? p : q));
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}

void test_conditional_lvalue_trylock_result(bool c) {
  int r1 = trylock_zero(), r2 = 1;
  set_int(c ? r1 : r2);
  if (r1 == 0) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }
}

void test_comma_lvalue_addr_trylock_result() {
  int r1 = trylock_zero();
  set_int_ptr(&(cond(), r1));
  if (r1 == 0) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
  }
}

// A variable cannot be modified through a const reference that is bound to
// it, nor by a call that it is passed to by const reference or by pointer to
// const. The reference itself changes with the variable and is not tracked.
void observe_cref(const int &);
void observe_cptr(const int *);
void fill_out(int *);
int *fill_out_and_return(int *);

void test_const_ref_binding() {
  int r = trylock(&mu);
  const int &ref = r;
  (void)ref;
  if (r) {
    data = 1;
    unlock(&mu);
  }
}

void test_const_ref_binding_alias() {
  int r = trylock(&mu);
  const int &ref = r;
  r = 0;
  if (ref) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_const_ref_and_const_ptr_args() {
  int r = trylock(&mu);
  observe_cref(r);
  observe_cptr(&r);
  if (r) {
    data = 1;
    unlock(&mu);
  }
}

// A variable whose address escapes may be modified through an alias by every
// statement that runs unknown code or modifies an object through a pointer or
// reference, until it is assigned a new value directly. It cannot be modified
// through an alias before its address escapes or before such a statement.
void test_out_param_after_branch() {
  int r = trylock(&mu);
  if (r) {
    data = 1;
    unlock(&mu);
  }
  fill_out(&r);
}

void test_out_param_before_branch() {
  int r = trylock(&mu);
  fill_out(&r);
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_out_param_later_in_loop(int n) {
  int r = trylock(&mu);
  for (int i = 0; i < n; ++i) {
    if (r) {
      data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
      unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
    }
    fill_out(&r);
  }
}

void test_out_param_returned_after_branch() {
  int r = trylock(&mu);
  if (r) {
    data = 1;
    unlock(&mu);
  }
  *fill_out_and_return(&r) = 0;
}

void test_out_param_in_loop_after_branch(int n) {
  for (int i = 0; i < n; ++i) {
    int r = trylock(&mu);
    if (r) {
      data = 1;
      unlock(&mu);
    }
    fill_out(&r);
  }
}

void test_reassign_after_out_param() {
  int r = 0;
  fill_out(&r);
  r = trylock(&mu);
  if (r) {
    data = 1;
    unlock(&mu);
  }
}

void test_reassign_after_out_param_then_call() {
  int r = 0;
  fill_out(&r);
  r = trylock(&mu);
  cond();
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_copy_before_out_param() {
  int r = trylock(&mu);
  bool ok = r != 0;
  fill_out(&r);
  if (ok) {
    data = 1;
    unlock(&mu);
  }
}

void test_copy_before_conditional_out_param(bool c) {
  int r = trylock(&mu);
  bool ok = r;
  if (c)
    fill_out(&r);
  if (ok) {
    data = 1;
    unlock(&mu);
  }
}

struct Status {
  int code;
};
Status fill_out_status(int *);
struct Mutator {
  Mutator(int &);
};

void test_escapes_after_branch() {
  int r = trylock(&mu);
  if (r) {
    data = 1;
    unlock(&mu);
  }
  Mutator m(r);
  Status s = fill_out_status(&r);
  int *p = &r;
  *p = 0;
  auto lam = [&r]() { r = 0; };
  lam();
}

void test_store_through_pointer() {
  int r = 0;
  int *p = &r;
  r = trylock(&mu);
  *p = 0;
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_store_through_pointer_on_one_path(bool c) {
  int r = trylock(&mu);
  int *p = &r;
  if (c)
    *p = 0;
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_store_through_pointer_in_loop(int n) {
  int r = trylock(&mu);
  int *p = &r;
  for (int i = 0; i < n; ++i) {
    if (r) {
      data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
      unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
    }
    *p = 0;
  }
}

void test_ref_capture_called() {
  int r = 0;
  auto lam = [&r]() { r = 0; };
  r = trylock(&mu);
  lam();
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void test_nonconst_ref_binding() {
  int r = trylock(&mu);
  int &ref = r;
  r = 0;
  if (ref) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

struct KeepAddr {
  KeepAddr(int *);
  ~KeepAddr();
};

void test_destructor_after_escape() {
  int r = 0;
  {
    KeepAddr k(&r);
    r = trylock(&mu);
  }
  if (r) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(&mu); // expected-warning{{releasing mutex 'mu' that was not held}}
  }
}

void observe_cptr_ref(const int *const &);

void test_const_pointee_args(bool c) {
  int r1 = trylock(&mu), r2 = 0;
  observe_cptr_ref(&r1);
  observe_cptr(c ? &r1 : &r2);
  if (r1) {
    data = 1;
    unlock(&mu);
  }
}

#if __cplusplus > 202002L
// The object argument of a call of a static call operator is not passed to
// the operator.
struct StaticRedirect {
  static void operator()(Mutex *&p, const int &) { p = &mu_ctor_other; }
};
void test_static_call_operator_mutates_local_ptr(StaticRedirect redir) {
  Mutex *p = &mu;
  redir(p, 0);
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}

void test_static_lambda_mutates_local_ptr() {
  auto redir = [](Mutex *&p, const int &) static { p = &mu_ctor_other; };
  Mutex *p = &mu;
  redir(p, 0);
  if (trylock(p)) {
    data = 1; // expected-warning{{writing variable 'data' requires holding mutex 'mu' exclusively}}
    unlock(p);
  }
}
#endif
