// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:     misc-const-correctness.AnalyzeParameters: true, \
// RUN:     misc-const-correctness.WarnPointersAsValues: true, \
// RUN:     misc-const-correctness.TransformPointersAsValues: true \
// RUN:   }}" -- -I %S/Inputs/const-correctness -fno-delayed-template-parsing

struct Bar {
  void const_method() const;
  void mutating_method();
  int value;
};

void ref_param_const(Bar& b);
// CHECK-FIXES: void ref_param_const(Bar const& b);

void ref_param_const(Bar& b) {
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void ref_param_const(Bar const& b) {
  b.const_method();
}

void ref_param_already_const(const Bar& f) {
  f.const_method();
}

void ref_param_mutated(Bar& f) {
  f.mutating_method();
}

void ref_param_member_modified(Bar& b) {
  b.value = 42;
}

void pointer_param_read_only(Bar* b) {
  // CHECK-MESSAGES: [[@LINE-1]]:30: warning: pointee of variable 'b' of type 'Bar *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:30: warning: variable 'b' of type 'Bar *' can be declared 'const'
  // CHECK-FIXES: void pointer_param_read_only(Bar const* const b) {
  b->const_method();
}

void pointer_param_mutated_pointee(Bar* b) {
  b->mutating_method();
}

void pointer_param_mutated_pointer(Bar* b) {
  // CHECK-MESSAGES: [[@LINE-1]]:36: warning: pointee of variable 'b' of type 'Bar *' can be declared 'const'
  // CHECK-FIXES: void pointer_param_mutated_pointer(Bar const* b) {
  b = nullptr;
}

void value_param_int(int x) {
  int y = x + 1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'y' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const y = x + 1;
}

void value_param_struct(Bar b) {
  b.const_method();
}

void multiple_params_mixed(int x, Bar& b, Bar& f) {
  // CHECK-MESSAGES: [[@LINE-1]]:35: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void multiple_params_mixed(int x, Bar const& b, Bar& f) {
  int y = x;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'y' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const y = x;
  b.const_method();
  f.mutating_method();
}

void rvalue_ref_param(Bar&& b) {
  b.const_method();
}

void pass_to_const_ref(const Bar& b);
void pass_to_nonconst_ref(Bar& b);

void param_passed_to_const_ref(Bar& b) {
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void param_passed_to_const_ref(Bar const& b) {
  pass_to_const_ref(b);
}

void param_passed_to_nonconst_ref(Bar& b) {
  pass_to_nonconst_ref(b);
}

template<typename T>
void template_param(T& t) {
  t.const_method();
}

template<typename T>
void forwarding_ref(T&& t) {
  t.mutating_method();
}

template<typename T>
void specialized_func(T& t) {
  t.const_method();
}

template<>
void specialized_func<Bar>(Bar& b) {
  b.const_method();
}

template<int N>
void non_type_template_param(Bar& b) {
  b.const_method();
}

template<typename... Args>
void variadic_template(Args&... args) {
  (args.const_method(), ...);
}

template<typename First, typename... Rest>
void variadic_first_param(First& first, Rest&... rest) {
  first.const_method();
}

template<typename T>
struct is_bar {
  static constexpr bool value = false;
};

template<>
struct is_bar<Bar> {
  static constexpr bool value = true;
};

template<typename T, typename = void>
struct enable_if {};

template<typename T>
struct enable_if<T, typename T::type> {
  using type = typename T::type;
};

struct true_type { using type = void; };
struct false_type {};

template<bool B>
struct bool_constant : false_type {};

template<>
struct bool_constant<true> : true_type {};

template<typename T>
void sfinae_func(T& t, typename enable_if<bool_constant<is_bar<T>::value>>::type* = nullptr) {
  t.const_method();
}

void instantiate() {
  int a = 42;
  Bar b;
  template_param(b);
  forwarding_ref(b);
  specialized_func(b);
  non_type_template_param<2>(b);
  variadic_template(b, b);
  variadic_first_param(b, a);
  sfinae_func(b);
}

// Leave this for further reference if const-correctness is implemented on template functions/methods 

template<typename T>
struct TemplateClass {
  void non_template_method(Bar& b) {
    b.const_method();
  }

  template<typename U>
  void template_method(U& u) {
    u.const_method();
  }

  static void static_method(Bar& b) {
    b.const_method();
  }
};

template struct TemplateClass<int>;

template<typename Outer>
struct OuterTemplate {
  template<typename Inner>
  static void nested_template_func(Outer& o, Inner& i) {
    o.const_method();
    i.const_method();
  }
};

template<typename T>
using RefAlias = T&;

template<typename T>
void alias_template_param(RefAlias<T> t) {
  t.const_method();
}

template<typename T>
void requires_const_method(T& t) {
}

void use_requires_const_method() {
  Bar b;
  requires_const_method(b);
}

void lambda_params() {
  auto lambda = [](Bar& b) {
    b.const_method();
  };
}

struct Container {
  Container(int* x) {}
  void member_func(Bar& b) {
    b.const_method();
  }
};

void builtin_int_ref(int& x) {
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'x' of type 'int &' can be declared 'const'
  // CHECK-FIXES: void builtin_int_ref(int const& x) {
  int y = x;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'y' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const y = x;
}

void builtin_int_ref_mutated(int& x) {
  x = 42;
}

void decl_multiple(Bar& b);
// CHECK-FIXES: void decl_multiple(Bar const& b);
void decl_multiple(Bar& b);
// CHECK-FIXES: void decl_multiple(Bar const& b);

void decl_multiple(Bar& b) {
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void decl_multiple(Bar const& b) {
  b.const_method();
}

void decl_multi_params(int& x, Bar& b, Bar& f);
// CHECK-FIXES: void decl_multi_params(int const& x, Bar const& b, Bar& f);

void decl_multi_params(int& x, Bar& b, Bar& f) {
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: variable 'x' of type 'int &' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:32: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void decl_multi_params(int const& x, Bar const& b, Bar& f) {
  int y = x;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'y' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const y = x;
  b.const_method();
  f.mutating_method();  // f is mutated
}

namespace ns {
  void namespaced_func(Bar& f);
  // CHECK-FIXES: void namespaced_func(Bar const& f);

  void namespaced_func(Bar& f) {
    // CHECK-MESSAGES: [[@LINE-1]]:24: warning: variable 'f' of type 'Bar &' can be declared 'const'
    // CHECK-FIXES: void namespaced_func(Bar const& f) {
    f.const_method();
  }
}

using int_ptr = int*;
using f_signature = void(int*);

void decl_different_style(int_ptr);
f_signature decl_different_style;
void decl_different_style(int* p) {
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:27: warning: variable 'p' of type 'int *' can be declared 'const'
  // No CHECK-FIXES - declaration uses 'using'
}

typedef int* int_ptr_typedef;
void typedef_decl(int_ptr_typedef);
void typedef_decl(int* p) {
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:19: warning: variable 'p' of type 'int *' can be declared 'const'
  // No CHECK-FIXES - declaration uses 'typedef'
}

void multi_param_one_alias(int_ptr, int*);
void multi_param_one_alias(int* p, int* q) {
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:28: warning: variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-3]]:36: warning: pointee of variable 'q' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-4]]:36: warning: variable 'q' of type 'int *' can be declared 'const'
  // CHECK-FIXES: void multi_param_one_alias(int_ptr, int const*const );
  // CHECK-FIXES-NEXT: void multi_param_one_alias(int* p, int const* const q) {
}

void func_ptr_param(void (*fp)(int&)) {
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: variable 'fp' of type 'void (*)(int &)' can be declared 'const'
  // CHECK-FIXES: void func_ptr_param(void (*const fp)(int&)) {
  int x = 5;
  fp(x);
}

void func_ptr_param_reassigned(void (*fp)(int&)) {
  int x = 5;
  fp(x);
  fp = nullptr;
}

struct Block {
  void render(int& x) const { x = 0; }
};

void member_ptr_param(void (Block::*mp)(int&) const) {
}

namespace std {
  template<typename T> class function;
  template<typename R, typename... Args>
  class function<R(Args...)> {
  public:
    template<typename F> function(F&&) {}
    R operator()(Args...) const { return R(); }
  };
}

struct Decl {};

void std_function_param(std::function<void(Decl*)> callback) {
  Decl d;
  callback(&d);
  std::function<void(Decl*)> const cb = nullptr;
}

void std_function_ref_param(std::function<void(int&)> callback) {
  int x = 5;
  callback(x);
}

void double_ptr_inner_not_modified(int** pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:36: warning: pointee of variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:36: warning: variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-FIXES: void double_ptr_inner_not_modified(int* const* const pp) {
  int* p = *pp;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'p' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int const* const p = *pp;
}

void double_ptr_inner_modified(int** pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-FIXES: void double_ptr_inner_modified(int** const pp) {
  *pp = nullptr;
}

void double_ptr_outer_not_modified(int** pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:36: warning: pointee of variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:36: warning: variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-FIXES: void double_ptr_outer_not_modified(int* const* const pp) {
  int* local = *pp;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'local' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'local' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int const* const local = *pp;
}

void triple_ptr_read_only(int*** ppp) {
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: pointee of variable 'ppp' of type 'int ***' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:27: warning: variable 'ppp' of type 'int ***' can be declared 'const'
  // CHECK-FIXES: void triple_ptr_read_only(int** const* const ppp) {
  int** pp = *ppp;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-FIXES: int* const* const pp = *ppp;
}

void triple_ptr_middle_modified(int*** ppp) {
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: pointee of variable 'ppp' of type 'int ***' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:33: warning: variable 'ppp' of type 'int ***' can be declared 'const'
  // CHECK-FIXES: void triple_ptr_middle_modified(int** const* const ppp) {
  **ppp = nullptr;
}

void ref_to_ptr_both_readonly(int*& p) {
  // CHECK-MESSAGES: [[@LINE-1]]:31: warning: variable 'p' of type 'int *&' can be declared 'const'
  // CHECK-FIXES: void ref_to_ptr_both_readonly(int* const& p) {
  int val = *p;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'val' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const val = *p;
}

void ref_to_ptr_pointer_modified(int*& p) {
  p = nullptr;
}

void ref_to_ptr_pointee_modified(int*& p) {
  // CHECK-MESSAGES: [[@LINE-1]]:34: warning: variable 'p' of type 'int *&' can be declared 'const'
  // CHECK-FIXES: void ref_to_ptr_pointee_modified(int* const& p) {
  *p = 42;
}

void ref_to_const_ptr(int* const& p) {
  int val = *p;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'val' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const val = *p;
}

void const_ref_to_ptr(int* const& p) {
  *p = 42;  // Pointee modification is allowed
}

void ref_to_ptr_to_const(const int*& p) {
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: variable 'p' of type 'const int *&' can be declared 'const'
  // CHECK-FIXES: void ref_to_ptr_to_const(const int* const& p) {
  int val = *p;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'val' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const val = *p;
}

void ref_to_ptr_to_const_modified(const int*& p) {
  p = nullptr;
}

void double_ptr_const_inner(const int** pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: pointee of variable 'pp' of type 'const int **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:29: warning: variable 'pp' of type 'const int **' can be declared 'const'
  // CHECK-FIXES: void double_ptr_const_inner(const int* const* const pp) {
  const int* p = *pp;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p' of type 'const int *' can be declared 'const'
  // CHECK-FIXES: const int* const p = *pp;
}

void double_ptr_modify_value(int** pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:30: warning: pointee of variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:30: warning: variable 'pp' of type 'int **' can be declared 'const'
  // CHECK-FIXES: void double_ptr_modify_value(int* const* const pp) {
  **pp = 42;
}

void ref_to_double_ptr(int**& pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: variable 'pp' of type 'int **&' can be declared 'const'
  // CHECK-FIXES: void ref_to_double_ptr(int** const& pp) {
  int* p = *pp;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'p' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int const* const p = *pp;
}

void ref_to_double_ptr_outer_modified(int**& pp) {
  pp = nullptr;
}

void ref_to_double_ptr_inner_modified(int**& pp) {
  // CHECK-MESSAGES: [[@LINE-1]]:39: warning: variable 'pp' of type 'int **&' can be declared 'const'
  // CHECK-FIXES: void ref_to_double_ptr_inner_modified(int** const& pp) {
  *pp = nullptr;
}

void array_of_ptrs(int* arr[]) {
  int* p = arr[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'p' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int const* const p = arr[0];
}

void ptr_to_array(int (*arr)[10]) {
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: pointee of variable 'arr' of type 'int (*)[10]' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:19: warning: variable 'arr' of type 'int (*)[10]' can be declared 'const'
  // CHECK-FIXES: void ptr_to_array(int  const(*const arr)[10]) {
  int val = (*arr)[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'val' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const val = (*arr)[0];
}

void ptr_to_array_modified(int (*arr)[10]) {
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: variable 'arr' of type 'int (*)[10]' can be declared 'const'
  // CHECK-FIXES: void ptr_to_array_modified(int (*const arr)[10]) {
  (*arr)[0] = 42;
}

void ref_to_ptr_array(int* (&arr)[5]) {
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: variable 'arr' of type 'int *(&)[5]' can be declared 'const'
  // CHECK-FIXES: void ref_to_ptr_array(int*  const(&arr)[5]) {
  int* p = arr[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p' of type 'int *' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: variable 'p' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int const* const p = arr[0];
}

void ref_to_ptr_array_modified(int* (&arr)[5]) {
  arr[0] = nullptr;
}

void struct_ptr_param(Bar** bp) {
  // CHECK-MESSAGES: [[@LINE-1]]:23: warning: pointee of variable 'bp' of type 'Bar **' can be declared 'const'
  // CHECK-MESSAGES: [[@LINE-2]]:23: warning: variable 'bp' of type 'Bar **' can be declared 'const'
  // CHECK-FIXES: void struct_ptr_param(Bar* const* const bp) {
  (*bp)->const_method();
}

void struct_ptr_param_modified(Bar** bp) {
  // CHECK-MESSAGES: [[@LINE-1]]:32: warning: variable 'bp' of type 'Bar **' can be declared 'const'
  // CHECK-FIXES: void struct_ptr_param_modified(Bar** const bp) {
  (*bp)->mutating_method();
}
