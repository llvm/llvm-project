// RUN: %check_clang_tidy %s cppcoreguidelines-const-correctness %t -- \
// RUN:   -config="{CheckOptions: [\
// RUN:   {key: 'cppcoreguidelines-const-correctness.TransformValues', value: 1},\
// RUN:   {key: 'cppcoreguidelines-const-correctness.WarnPointersAsValues', value: 0}, \
// RUN:   {key: 'cppcoreguidelines-const-correctness.TransformPointersAsValues', value: 0}, \
// RUN:   ]}" --

bool global;
char np_global = 0; // globals can't be known to be const

namespace foo {
int scoped;
float np_scoped = 1; // namespace variables are like globals
} // namespace foo

// Lambdas should be ignored, because they do not follow the normal variable
// semantic (e.g. the type is only known to the compiler).
void lambdas() {
  auto Lambda = [](int i) { return i < 0; };
}

void some_function(double, wchar_t);

void some_function(double np_arg0, wchar_t np_arg1) {
  int p_local0 = 2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  // CHECK-FIXES: const
}

void nested_scopes() {
  {
    int p_local1 = 42;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: variable 'p_local1' of type 'int' can be declared 'const'
    // CHECK-FIXES: const
  }
}

template <typename T>
void define_locals(T np_arg0, T &np_arg1, int np_arg2) {
  T np_local0 = 0;
  int p_local1 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
  // CHECK-FIXES: const
}

void template_instantiation() {
  const int np_local0 = 42;
  int np_local1 = 42;

  define_locals(np_local0, np_local1, np_local0);
  define_locals(np_local1, np_local1, np_local1);
}

struct ConstNonConstClass {
  ConstNonConstClass();
  ConstNonConstClass(double &np_local0);
  double nonConstMethod() {}
  double constMethod() const {}
  double modifyingMethod(double &np_arg0) const;

  double NonConstMember;
  const double ConstMember;

  double &NonConstMemberRef;
  const double &ConstMemberRef;

  double *NonConstMemberPtr;
  const double *ConstMemberPtr;
};

void direct_class_access() {
  ConstNonConstClass p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'ConstNonConstClass' can be declared 'const'
  // CHECK-FIXES: const
  p_local0.constMethod();
}

void class_access_array() {
  ConstNonConstClass p_local0[2];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'ConstNonConstClass [2]' can be declared 'const'
  // CHECK-FIXES: const
  p_local0[0].constMethod();
}

struct MyVector {
  double *begin();
  const double *begin() const;

  double *end();
  const double *end() const;

  double &operator[](int index);
  double operator[](int index) const;

  double values[100];
};

void vector_usage() {
  double p_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double [10]' can be declared 'const'
  // CHECK-FIXES: const
}

void range_for() {
  int np_local0[2] = {1, 2};
  int *np_local3[2] = {&np_local0[0], &np_local0[1]};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'np_local3' of type 'int *[2]' can be declared 'const'
  // CHECK-FIXES: const
  for (int *non_const_ptr : np_local3) {
    *non_const_ptr = 45;
  }
}

void casts() {
  decltype(sizeof(void *)) p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'decltype(sizeof(void *))' (aka 'unsigned long') can be declared 'const'
  // CHECK-FIXES: const
}

// taken from http://www.cplusplus.com/reference/type_traits/integral_constant/
template <typename T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant<T, v>;
  constexpr operator T() { return v; }
};

template <typename T>
struct is_integral : integral_constant<bool, false> {};
template <>
struct is_integral<int> : integral_constant<bool, true> {};

template <typename T>
struct not_integral : integral_constant<bool, false> {};
template <>
struct not_integral<double> : integral_constant<bool, true> {};

// taken from http://www.cplusplus.com/reference/type_traits/enable_if/
template <bool Cond, typename T = void>
struct enable_if {};

template <typename T>
struct enable_if<true, T> { using type = T; };

template <typename T>
struct TMPClass {
  T alwaysConst() const { return T{}; }

  template <typename T2 = T, typename = typename enable_if<is_integral<T2>::value>::type>
  T sometimesConst() const { return T{}; }

  template <typename T2 = T, typename = typename enable_if<not_integral<T2>::value>::type>
  T sometimesConst() { return T{}; }
};

void meta_type() {
  TMPClass<int> p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'TMPClass<int>' can be declared 'const'
  // CHECK-FIXES: const
  p_local0.alwaysConst();
  p_local0.sometimesConst();

  TMPClass<double> p_local1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'TMPClass<double>' can be declared 'const'
  // CHECK-FIXES: const
  p_local1.alwaysConst();
}
