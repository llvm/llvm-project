// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s -Wno-undefined-bool-conversion

typedef __INTPTR_TYPE__ intptr_t;

template <typename T>
void clang_analyzer_dump(T x);

const int& g() {
  int s;
  return s; // expected-warning{{Address of stack memory associated with local variable 's' returned}} expected-warning{{reference to stack memory associated with local variable 's' returned}}
}

const int& g2() {
  int s1;
  int &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  return s2; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{reference to stack memory associated with local variable 's1' returned}}
}

const int& g3() {
  int s1;
  int &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  int &s3 = s2; // expected-note {{binding reference variable 's3' here}}
  return s3; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{reference to stack memory associated with local variable 's1' returned}}
}

void g4() {
  static const int &x = 3; // no warning
}

int get_value();

const int &get_reference1() { return get_value(); } // expected-warning{{Address of stack memory associated with temporary object of type 'int' returned}} expected-warning {{returning reference to local temporary}}

const int &get_reference2() {
  const int &x = get_value(); // expected-note {{binding reference variable 'x' here}}
  return x; // expected-warning{{Address of stack memory associated with temporary object of type 'int' lifetime extended by local variable 'x' returned to caller}} expected-warning {{returning reference to local temporary}} 
}

const int &get_reference3() {
  const int &x1 = get_value(); // expected-note {{binding reference variable 'x1' here}}
  const int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return x2; // expected-warning{{Address of stack memory associated with temporary object of type 'int' lifetime extended by local variable 'x1' returned to caller}} expected-warning {{returning reference to local temporary}}
}

int global_var;
int *f1() {
  int &y = global_var;
  return &y;
}

int *f2() {
  int x1;
  int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return &x2; // expected-warning{{Address of stack memory associated with local variable 'x1' returned}} expected-warning {{address of stack memory associated with local variable 'x1' returned}}
}

int *f3() {
  int x1;
  int *const &x2 = &x1; // expected-note {{binding reference variable 'x2' here}}
  return x2; // expected-warning {{address of stack memory associated with local variable 'x1' returned}} expected-warning {{Address of stack memory associated with local variable 'x1' returned to caller}}
}

const int *f4() {
  const int &x1 = get_value(); // expected-note {{binding reference variable 'x1' here}}
  const int &x2 = x1; // expected-note {{binding reference variable 'x2' here}}
  return &x2; // expected-warning{{Address of stack memory associated with temporary object of type 'int' lifetime extended by local variable 'x1' returned to caller}} expected-warning {{returning address of local temporary}}
}

struct S {
  int x;
};

int *mf() {
  S s1;
  S &s2 = s1; // expected-note {{binding reference variable 's2' here}}
  int &x = s2.x; // expected-note {{binding reference variable 'x' here}}
  return &x; // expected-warning{{Address of stack memory associated with local variable 's1' returned}} expected-warning {{address of stack memory associated with local variable 's1' returned}}
}

void *lf() {
    label:
    void *const &x = &&label; // expected-note {{binding reference variable 'x' here}}
    return x; // expected-warning {{returning address of label, which is local}}
}

template <typename T>
struct TS {
  int *get();
  int *m() {
    int *&x = get();
    return x;
  }
};

int* f5() {
  int& i = i; // expected-warning {{Assigned value is garbage or undefined}} expected-warning{{reference 'i' is not yet bound to a value when used within its own initialization}}
  return &i;
}

void *radar13226577() {
    void *p = &p;
    return p; // expected-warning {{stack memory associated with local variable 'p' returned to caller}}
}

namespace rdar13296133 {
  class ConvertsToBool {
  public:
    operator bool() const { return this; }
  };

  class ConvertsToIntptr {
  public:
    operator intptr_t() const { return reinterpret_cast<intptr_t>(this); }
  };

  class ConvertsToPointer {
  public:
    operator const void *() const { return this; }
  };

  intptr_t returnAsNonLoc() {
    ConvertsToIntptr obj;
    return obj; // expected-warning{{Address of stack memory associated with local variable 'obj' returned to caller}}
  }

  bool returnAsBool() {
    ConvertsToBool obj;
    return obj; // no-warning
  }

  intptr_t returnAsNonLocViaPointer() {
    ConvertsToPointer obj;
    return reinterpret_cast<intptr_t>(static_cast<const void *>(obj)); // expected-warning{{Address of stack memory associated with local variable 'obj' returned to caller}}
  }

  bool returnAsBoolViaPointer() {
    ConvertsToPointer obj;
    return obj; // no-warning
  }
} // namespace rdar13296133

void write_stack_address_to(char **q) {
  char local;
  *q = &local;
  // expected-warning@-1 {{Address of stack memory associated with local \
variable 'local' is still referred to by the caller variable 'p' upon \
returning to the caller}}
}

void test_stack() {
  char *p;
  write_stack_address_to(&p);
}

struct C {
  ~C() {} // non-trivial class
};

C make1() {
  C c;
  return c; // no-warning
}

void test_copy_elision() {
  C c1 = make1();
}

namespace leaking_via_direct_pointer {
void* returned_direct_pointer_top() {
  int local = 42;
  int* p = &local;
  return p; // expected-warning{{associated with local variable 'local' returned}}
}

int* returned_direct_pointer_callee() {
  int local = 42;
  int* p = &local;
  return p; // expected-warning{{associated with local variable 'local' returned}}
}

void returned_direct_pointer_caller() {
  int* loc_ptr = nullptr;
  loc_ptr = returned_direct_pointer_callee();
  (void)loc_ptr;
}

void* global_ptr;

void global_direct_pointer() {
  int local = 42;
  global_ptr = &local; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_ptr'}}
}

void static_direct_pointer_top() {
  int local = 42;
  static int* p = &local;
  (void)p; // expected-warning{{local variable 'local' is still referred to by the static variable 'p'}}
}

void static_direct_pointer_callee() {
  int local = 42;
  static int* p = &local;
  (void)p; // expected-warning{{local variable 'local' is still referred to by the static variable 'p'}}
}

void static_direct_pointer_caller() {
  static_direct_pointer_callee();
}

void lambda_to_global_direct_pointer() {
  auto lambda = [&] {
    int local = 42;
    global_ptr = &local; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_ptr'}}
  };
  lambda();
}

void lambda_to_context_direct_pointer() {
  int *p = nullptr;
  auto lambda = [&] {
    int local = 42;
    p = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'p'}}
  };
  lambda();
  (void)p;
}

template<typename Callable>
class MyFunction {
  Callable* fptr;
  public:
  MyFunction(Callable* callable) :fptr(callable) {}
};

void* lambda_to_context_direct_pointer_uncalled() {
  int *p = nullptr;
  auto lambda = [&] {
    int local = 42;
    p = &local; // no-warning: analyzed only as top-level, ignored explicitly by the checker
  };
  return new MyFunction(&lambda);
}

void lambda_to_context_direct_pointer_lifetime_extended() {
  int *p = nullptr;
  auto lambda = [&] {
    int&& local = 42;
    p = &local; // expected-warning{{'int' lifetime extended by local variable 'local' is still referred to by the caller variable 'p'}}
  };
  lambda();
  (void)p;
}

template<typename Callback>
void lambda_param_capture_direct_pointer_callee(Callback& callee) {
  int local = 42;
  callee(local); // expected-warning{{'local' is still referred to by the caller variable 'p'}}
}

void lambda_param_capture_direct_pointer_caller() {
  int* p = nullptr;
  auto capt = [&p](int& param) {
    p = &param;
  };
  lambda_param_capture_direct_pointer_callee(capt);
}
} // namespace leaking_via_direct_pointer

namespace leaking_via_ptr_to_ptr {
void** returned_ptr_to_ptr_top() {
  int local = 42;
  int* p = &local;
  void** pp = (void**)&p;
  return pp; // expected-warning{{associated with local variable 'p' returned}}
}

void** global_pp;

void global_ptr_local_to_ptr() {
  int local = 42;
  int* p = &local;
  global_pp = (void**)&p; // expected-warning{{local variable 'p' is still referred to by the global variable 'global_pp'}}
}

void global_ptr_to_ptr() {
  int local = 42;
  *global_pp = &local; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_pp'}}
}

void *** global_ppp;

void global_ptr_to_ptr_to_ptr() {
  int local = 42;
  **global_ppp = &local; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_ppp'}}
}

void** get_some_pp();

void static_ptr_to_ptr() {
  int local = 42;
  static void** pp = get_some_pp();
  *pp = &local;
} // no-warning False Negative, requires relating multiple bindings to cross the invented pointer.

void param_ptr_to_ptr_top(void** pp) {
  int local = 42;
  *pp = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'pp'}}
}

void param_ptr_to_ptr_callee(void** pp) {
  int local = 42;
  *pp = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'p'}}
}

void param_ptr_to_ptr_caller() {
  void* p = nullptr;
  param_ptr_to_ptr_callee((void**)&p);
}

void param_ptr_to_ptr_to_ptr_top(void*** ppp) {
  int local = 42;
  **ppp = &local; // expected-warning {{local variable 'local' is still referred to by the caller variable 'ppp'}}
}

void param_ptr_to_ptr_to_ptr_callee(void*** ppp) {
  int local = 42;
  **ppp = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'pp'}}
}

void param_ptr_to_ptr_to_ptr_caller(void** pp) {
  param_ptr_to_ptr_to_ptr_callee(&pp);
}

void lambda_to_context_ptr_to_ptr(int **pp) {
  auto lambda = [&] {
    int local = 42;
    *pp = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'pp'}}
  };
  lambda();
  (void)*pp;
}

void param_ptr_to_ptr_fptr(int **pp) {
  int local = 42;
  *pp = &local; // expected-warning{{local variable 'local' is still referred to by the caller variable 'p'}}
}

void param_ptr_to_ptr_fptr_caller(void (*fptr)(int**)) {
  int* p = nullptr;
  fptr(&p);
}

void param_ptr_to_ptr_caller_caller() {
  void (*fptr)(int**) = param_ptr_to_ptr_fptr;
  param_ptr_to_ptr_fptr_caller(fptr);
}
} // namespace leaking_via_ptr_to_ptr

namespace leaking_via_ref_to_ptr {
void** make_ptr_to_ptr();
void*& global_rtp = *make_ptr_to_ptr();

void global_ref_to_ptr() {
  int local = 42;
  int* p = &local;
  global_rtp = p; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_rtp'}}
}

void static_ref_to_ptr() {
  int local = 42;
  static void*& p = *make_ptr_to_ptr();
  p = &local;
  (void)p;
} // no-warning False Negative, requires relating multiple bindings to cross the invented pointer.

void param_ref_to_ptr_top(void*& rp) {
  int local = 42;
  int* p = &local;
  rp = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'rp'}}
}

void param_ref_to_ptr_callee(void*& rp) {
  int local = 42;
  int* p = &local;
  rp = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'p'}}
}

void param_ref_to_ptr_caller() {
  void* p = nullptr;
  param_ref_to_ptr_callee(p);
}
} // namespace leaking_via_ref_to_ptr

namespace leaking_via_arr_of_ptr_static_idx {
void** returned_arr_of_ptr_top() {
  int local = 42;
  int* p = &local;
  void** arr = new void*[2];
  arr[1] = p;
  return arr;
} // no-warning False Negative

void** returned_arr_of_ptr_callee() {
  int local = 42;
  int* p = &local;
  void** arr = new void*[2];
  arr[1] = p;
  return arr;
} // no-warning False Negative

void returned_arr_of_ptr_caller() {
  void** arr = returned_arr_of_ptr_callee();
  (void)arr[1];
}

void* global_aop[2];

void global_arr_of_ptr() {
  int local = 42;
  int* p = &local;
  global_aop[1] = p; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_aop'}}
}

void static_arr_of_ptr() {
  int local = 42;
  static void* arr[2];
  arr[1] = &local;
  (void)arr[1]; // expected-warning{{local variable 'local' is still referred to by the static variable 'arr'}}
}

void param_arr_of_ptr_top(void* arr[2]) {
  int local = 42;
  int* p = &local;
  arr[1] = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'arr'}}
}

void param_arr_of_ptr_callee(void* arr[2]) {
  int local = 42;
  int* p = &local;
  arr[1] = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'arrStack'}}
}

void param_arr_of_ptr_caller() {
  void* arrStack[2];
  param_arr_of_ptr_callee(arrStack);
  (void)arrStack[1];
}
} // namespace leaking_via_arr_of_ptr_static_idx

namespace leaking_via_arr_of_ptr_dynamic_idx {
void** returned_arr_of_ptr_top(int idx) {
  int local = 42;
  int* p = &local;
  void** arr = new void*[2];
  arr[idx] = p;
  return arr;
} // no-warning False Negative

void** returned_arr_of_ptr_callee(int idx) {
  int local = 42;
  int* p = &local;
  void** arr = new void*[2];
  arr[idx] = p;
  return arr;
} // no-warning False Negative

void returned_arr_of_ptr_caller(int idx) {
  void** arr = returned_arr_of_ptr_callee(idx);
  (void)arr[idx];
}

void* global_aop[2];

void global_arr_of_ptr(int idx) {
  int local = 42;
  int* p = &local;
  global_aop[idx] = p; // expected-warning{{local variable 'local' is still referred to by the global variable 'global_aop'}}
}

void static_arr_of_ptr(int idx) {
  int local = 42;
  static void* arr[2];
  arr[idx] = &local;
  (void)arr[idx]; // expected-warning{{local variable 'local' is still referred to by the static variable 'arr'}}
}

void param_arr_of_ptr_top(void* arr[2], int idx) {
  int local = 42;
  int* p = &local;
  arr[idx] = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'arr'}}
}

void param_arr_of_ptr_callee(void* arr[2], int idx) {
  int local = 42;
  int* p = &local;
  arr[idx] = p; // expected-warning{{local variable 'local' is still referred to by the caller variable 'arrStack'}}
}

void param_arr_of_ptr_caller(int idx) {
  void* arrStack[2];
  param_arr_of_ptr_callee(arrStack, idx);
  (void)arrStack[idx];
}
} // namespace leaking_via_arr_of_ptr_dynamic_idx

namespace leaking_via_struct_with_ptr {
struct S {
  int* p;
};

S returned_struct_with_ptr_top() {
  int local = 42;
  S s;
  s.p = &local;
  return s;
} // no-warning False Negative, requires traversing returned LazyCompoundVals

S returned_struct_with_ptr_callee() {
  int local = 42;
  S s;
  s.p = &local;
  return s; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void returned_struct_with_ptr_caller() {
  S s = returned_struct_with_ptr_callee();
  (void)s.p;
}

S global_s;

void global_struct_with_ptr() {
  int local = 42;
  global_s.p = &local; // expected-warning{{'local' is still referred to by the global variable 'global_s'}}
}

void static_struct_with_ptr() {
  int local = 42;
  static S s;
  s.p = &local;
  (void)s.p; // expected-warning{{'local' is still referred to by the static variable 's'}}
}
} // namespace leaking_via_struct_with_ptr

namespace leaking_via_ref_to_struct_with_ptr {
struct S {
  int* p;
};

S &global_s = *(new S);

void global_ref_to_struct_with_ptr() {
  int local = 42;
  global_s.p = &local; // expected-warning{{'local' is still referred to by the global variable 'global_s'}}
}

void static_ref_to_struct_with_ptr() {
  int local = 42;
  static S &s = *(new S);
  s.p = &local;
  (void)s.p;
} // no-warning False Negative, requires relating multiple bindings to cross a heap region.

void param_ref_to_struct_with_ptr_top(S &s) {
  int local = 42;
  s.p = &local; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void param_ref_to_struct_with_ptr_callee(S &s) {
  int local = 42;
  s.p = &local; // expected-warning{{'local' is still referred to by the caller variable 'sStack'}}
}

void param_ref_to_struct_with_ptr_caller() {
  S sStack;
  param_ref_to_struct_with_ptr_callee(sStack);
}

template<typename Callable>
void lambda_param_capture_callee(Callable& callee) {
  int local = 42;
  callee(local); // expected-warning{{'local' is still referred to by the caller variable 'p'}}
}

void lambda_param_capture_caller() {
  int* p = nullptr;
  auto capt = [&p](int& param) {
    p = &param;
  };
  lambda_param_capture_callee(capt);
}
} // namespace leaking_via_ref_to_struct_with_ptr

namespace leaking_via_ptr_to_struct_with_ptr {
struct S {
  int* p;
};

S* returned_ptr_to_struct_with_ptr_top() {
  int local = 42;
  S* s = new S;
  s->p = &local;
  return s;
} // no-warning False Negative

S* returned_ptr_to_struct_with_ptr_callee() {
  int local = 42;
  S* s = new S;
  s->p = &local;
  return s;
} // no-warning False Negative

void returned_ptr_to_struct_with_ptr_caller() {
  S* s = returned_ptr_to_struct_with_ptr_callee();
  (void)s->p;
}

S* global_s;

void global_ptr_to_struct_with_ptr() {
  int local = 42;
  global_s->p = &local; // expected-warning{{'local' is still referred to by the global variable 'global_s'}}
}

void static_ptr_to_struct_with_ptr_new() {
  int local = 42;
  static S* s = new S;
  s->p = &local;
  (void)s->p;
} // no-warning  False Negative, requires relating multiple bindings to cross a heap region.

S* get_some_s();

void static_ptr_to_struct_with_ptr_generated() {
  int local = 42;
  static S* s = get_some_s();
  s->p = &local;
} // no-warning False Negative, requires relating multiple bindings to cross the invented pointer.

void param_ptr_to_struct_with_ptr_top(S* s) {
  int local = 42;
  s->p = &local; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void param_ptr_to_struct_with_ptr_callee(S* s) {
  int local = 42;
  s->p = &local; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void param_ptr_to_struct_with_ptr_caller() {
  S s;
  param_ptr_to_struct_with_ptr_callee(&s);
  (void)s.p;
}
} // namespace leaking_via_ptr_to_struct_with_ptr

namespace leaking_via_arr_of_struct_with_ptr {
struct S {
  int* p;
};

S* returned_ptr_to_struct_with_ptr_top() {
  int local = 42;
  S* s = new S[2];
  s[1].p = &local;
  return s;
} // no-warning False Negative

S* returned_ptr_to_struct_with_ptr_callee() {
  int local = 42;
  S* s = new S[2];
  s[1].p = &local;
  return s;
} // no-warning  False Negative

void returned_ptr_to_struct_with_ptr_caller() {
  S* s = returned_ptr_to_struct_with_ptr_callee();
  (void)s[1].p;
}

S global_s[2];

void global_ptr_to_struct_with_ptr() {
  int local = 42;
  global_s[1].p = &local; // expected-warning{{'local' is still referred to by the global variable 'global_s'}}
}

void static_ptr_to_struct_with_ptr_new() {
  int local = 42;
  static S* s = new S[2];
  s[1].p = &local;
  (void)s[1].p;
}

S* get_some_s();

void static_ptr_to_struct_with_ptr_generated() {
  int local = 42;
  static S* s = get_some_s();
  s[1].p = &local;
} // no-warning False Negative, requires relating multiple bindings to cross the invented pointer.

void param_ptr_to_struct_with_ptr_top(S s[2]) {
  int local = 42;
  s[1].p = &local; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void param_ptr_to_struct_with_ptr_callee(S s[2]) {
  int local = 42;
  s[1].p = &local; // expected-warning{{'local' is still referred to by the caller variable 's'}}
}

void param_ptr_to_struct_with_ptr_caller() {
  S s[2];
  param_ptr_to_struct_with_ptr_callee(s);
  (void)s[1].p;
}
} // namespace leaking_via_arr_of_struct_with_ptr

namespace leaking_via_nested_and_indirect {
struct NestedAndTransitive {
  int** p;
  NestedAndTransitive* next[3];
};

NestedAndTransitive global_nat;

void global_nested_and_transitive() {
  int local = 42;
  *global_nat.next[2]->next[1]->p = &local; // expected-warning{{'local' is still referred to by the global variable 'global_nat'}}
}

void param_nested_and_transitive_top(NestedAndTransitive* nat) {
  int local = 42;
  *nat->next[2]->next[1]->p = &local; // expected-warning{{'local' is still referred to by the caller variable 'nat'}}
}

void param_nested_and_transitive_callee(NestedAndTransitive* nat) {
  int local = 42;
  *nat->next[2]->next[1]->p = &local; // expected-warning{{'local' is still referred to by the caller variable 'natCaller'}}
}

void param_nested_and_transitive_caller(NestedAndTransitive natCaller) {
  param_nested_and_transitive_callee(&natCaller);
}

} // namespace leaking_via_nested_and_indirect

namespace leaking_as_member {
class CRef {
  int& ref; // expected-note{{reference member declared here}}
  CRef(int x) : ref(x) {}
  // expected-warning@-1 {{binding reference member 'ref' to stack allocated parameter 'x'}}
};

class CPtr {
  int* ptr;
  void memFun(int x) {
    ptr = &x;
  }
};
} // namespace leaking_as_member

namespace origin_region_limitation {
void leaker(int ***leakerArg) {
    int local;
    clang_analyzer_dump(*leakerArg); // expected-warning{{&SymRegion{reg_$0<int ** arg>}}}
    // Incorrect message: 'arg', after it is reinitialized with value returned by 'tweak'
    // is no longer relevant.
    // The message must refer to 'original_arg' instead, but there is no easy way to
    // connect the SymRegion stored in 'original_arg' and 'original_arg' as variable.
    **leakerArg = &local; // expected-warning{{ 'local' is still referred to by the caller variable 'arg'}}
}

int **tweak();

void foo(int **arg) {
    int **original_arg = arg;
    arg = tweak();
    leaker(&original_arg);
}
} // namespace origin_region_limitation

namespace leaking_via_indirect_global_invalidated {
void** global_pp;
void opaque();
void global_ptr_to_ptr() {
  int local = 42;
  *global_pp = &local;
  opaque();
  *global_pp = nullptr;
}
} // namespace leaking_via_indirect_global_invalidated

namespace not_leaking_via_simple_ptr {
void simple_ptr(const char *p) {
  char tmp;
  p = &tmp; // no-warning
}

void ref_ptr(const char *&p) {
  char tmp;
  p = &tmp; // expected-warning{{variable 'tmp' is still referred to by the caller variable 'p'}}
}

struct S {
  const char *p;
};

void struct_ptr(S s) {
  char tmp;
  s.p = &tmp; // no-warning
}

void array(const char arr[2]) {
  char tmp;
  arr = &tmp; // no-warning
}

extern void copy(char *output, const char *input, unsigned size);
extern bool foo(const char *input);
extern void bar(char *output, unsigned count);
extern bool baz(char *output, const char *input);

void repo(const char *input, char *output) {
  char temp[64];
  copy(temp, input, sizeof(temp));

  char result[64];
  input = temp;
  if (foo(temp)) {
    bar(result, sizeof(result));
    input = result;
  }
  if (!baz(output, input)) {
    copy(output, input, sizeof(result));
  }
}
} // namespace not_leaking_via_simple_ptr

namespace early_reclaim_dead_limitation {
void foo();
void top(char **p) {
  char local;
  *p = &local;
  foo(); // no-warning FIXME: p binding is reclaimed before the function end
}
} // namespace early_reclaim_dead_limitation
