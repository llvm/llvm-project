// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:   misc-const-correctness.TransformValues: true,\
// RUN:   misc-const-correctness.WarnPointersAsValues: false, \
// RUN:   misc-const-correctness.TransformPointersAsValues: false} \
// RUN:   }" -- -fno-delayed-template-parsing

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
  // CHECK-FIXES: int const p_local0 = 2;
}

void nested_scopes() {
  {
    int p_local1 = 42;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: variable 'p_local1' of type 'int' can be declared 'const'
    // CHECK-FIXES: int const p_local1 = 42;
  }
}

template <typename T>
void define_locals(T np_arg0, T &np_arg1, int np_arg2) {
  T np_local0 = 0;
  int p_local1 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const p_local1 = 42;
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
  // CHECK-FIXES: ConstNonConstClass const p_local0;
  p_local0.constMethod();
}

void class_access_array() {
  ConstNonConstClass p_local0[2];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'ConstNonConstClass[2]' can be declared 'const'
  // CHECK-FIXES: ConstNonConstClass const p_local0[2];
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
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double[10]' can be declared 'const'
  // CHECK-FIXES: double const p_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
}

void range_for() {
  int np_local0[2] = {1, 2};
  // The transformation is not possible because the range-for-loop mutates the array content.
  int *const np_local1[2] = {&np_local0[0], &np_local0[1]};
  for (int *non_const_ptr : np_local1) {
    *non_const_ptr = 45;
  }

  int *np_local2[2] = {&np_local0[0], &np_local0[1]};
  for (int *non_const_ptr : np_local2) {
    *non_const_ptr = 45;
  }
}

void decltype_declaration() {
  decltype(sizeof(void *)) p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'decltype(sizeof(void *))'
  // CHECK-FIXES: decltype(sizeof(void *)) const p_local0 = 42;
}

// Taken from libcxx/include/type_traits and improved readability.
template <class Tp, Tp v>
struct integral_constant {
  static constexpr const Tp value = v;
  using value_type = Tp;
  using type = integral_constant;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <typename T>
struct is_integral : integral_constant<bool, false> {};
template <>
struct is_integral<int> : integral_constant<bool, true> {};

template <typename T>
struct not_integral : integral_constant<bool, false> {};
template <>
struct not_integral<double> : integral_constant<bool, true> {};

template <bool, typename Tp = void>
struct enable_if {};

template <typename Tp>
struct enable_if<true, Tp> { using type = Tp; };

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
  // CHECK-FIXES: TMPClass<int> const p_local0;
  p_local0.alwaysConst();
  p_local0.sometimesConst();

  TMPClass<double> p_local1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'TMPClass<double>' can be declared 'const'
  // CHECK-FIXES: TMPClass<double> const p_local1;
  p_local1.alwaysConst();

  TMPClass<double> p_local2; // Don't attempt to make this const
  p_local2.sometimesConst();
}
