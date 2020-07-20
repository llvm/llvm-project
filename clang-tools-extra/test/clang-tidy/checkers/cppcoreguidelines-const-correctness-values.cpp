// RUN: %check_clang_tidy %s cppcoreguidelines-const-correctness %t -- \
// RUN:   -config="{CheckOptions: [\
// RUN:   {key: 'cppcoreguidelines-const-correctness.TransformValues', value: 1}, \
// RUN:   {key: 'cppcoreguidelines-const-correctness.WarnPointersAsValues', value: 0}, \
// RUN:   {key: 'cppcoreguidelines-const-correctness.TransformPointersAsValues', value: 0}, \
// RUN:   ]}" --

// ------- Provide test samples for primitive builtins ---------
// - every 'p_*' variable is a 'potential_const_*' variable
// - every 'np_*' variable is a 'non_potential_const_*' variable

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

  int np_local0;
  const int np_local1 = 42;

  unsigned int np_local2 = 3;
  np_local2 <<= 4;

  int np_local3 = 4;
  ++np_local3;
  int np_local4 = 4;
  np_local4++;

  int np_local5 = 4;
  --np_local5;
  int np_local6 = 4;
  np_local6--;
}

void nested_scopes() {
  int p_local0 = 2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  int np_local0 = 42;

  {
    int p_local1 = 42;
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: variable 'p_local1' of type 'int' can be declared 'const'
    np_local0 *= 2;
  }
}

void ignore_reference_to_pointers() {
  int *np_local0 = nullptr;
  int *&np_local1 = np_local0;
}

void some_lambda_environment_capture_all_by_reference(double np_arg0) {
  int np_local0 = 0;
  int p_local0 = 1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'

  int np_local2;
  const int np_local3 = 2;

  // Capturing all variables by reference prohibits making them const.
  [&]() { ++np_local0; };

  int p_local1 = 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
}

void some_lambda_environment_capture_all_by_value(double np_arg0) {
  int np_local0 = 0;
  int p_local0 = 1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'

  int np_local1;
  const int np_local2 = 2;

  // Capturing by value has no influence on them.
  [=]() { (void)p_local0; };

  np_local0 += 10;
}

void function_inout_pointer(int *inout);
void function_in_pointer(const int *in);

void some_pointer_taking(int *out) {
  int np_local0 = 42;
  const int *const p0_np_local0 = &np_local0;
  int *const p1_np_local0 = &np_local0;

  int np_local1 = 42;
  const int *const p0_np_local1 = &np_local1;
  int *const p1_np_local1 = &np_local1;
  *p1_np_local0 = 43;

  int np_local2 = 42;
  function_inout_pointer(&np_local2);

  // Prevents const.
  int np_local3 = 42;
  out = &np_local3; // This returns and invalid address, its just about the AST

  int p_local1 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
  const int *const p0_p_local1 = &p_local1;

  int p_local2 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'int' can be declared 'const'
  function_in_pointer(&p_local2);
}

void function_inout_ref(int &inout);
void function_in_ref(const int &in);

void some_reference_taking() {
  int np_local0 = 42;
  const int &r0_np_local0 = np_local0;
  int &r1_np_local0 = np_local0;
  r1_np_local0 = 43;
  const int &r2_np_local0 = r1_np_local0;

  int np_local1 = 42;
  function_inout_ref(np_local1);

  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  const int &r0_p_local0 = p_local0;

  int p_local1 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
  function_in_ref(p_local1);
}

double *non_const_pointer_return() {
  double p_local0 = 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double' can be declared 'const'
  double np_local0 = 24.4;

  return &np_local0;
}

const double *const_pointer_return() {
  double p_local0 = 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double' can be declared 'const'
  double p_local1 = 24.4;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'double' can be declared 'const'
  return &p_local1;
}

double &non_const_ref_return() {
  double p_local0 = 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double' can be declared 'const'
  double np_local0 = 42.42;
  return np_local0;
}

const double &const_ref_return() {
  double p_local0 = 0.0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double' can be declared 'const'
  double p_local1 = 24.4;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'double' can be declared 'const'
  return p_local1;
}

double *&return_non_const_pointer_ref() {
  double *np_local0 = nullptr;
  return np_local0;
}

void overloaded_arguments(const int &in);
void overloaded_arguments(int &inout);
void overloaded_arguments(const int *in);
void overloaded_arguments(int *inout);

void function_calling() {
  int np_local0 = 42;
  overloaded_arguments(np_local0);

  const int np_local1 = 42;
  overloaded_arguments(np_local1);

  int np_local2 = 42;
  overloaded_arguments(&np_local2);

  const int np_local3 = 42;
  overloaded_arguments(&np_local3);
}

template <typename T>
void define_locals(T np_arg0, T &np_arg1, int np_arg2) {
  T np_local0 = 0;
  np_local0 += np_arg0 * np_arg1;

  T np_local1 = 42;
  np_local0 += np_local1;

  // Used as argument to an overloaded function with const and non-const.
  T np_local2 = 42;
  overloaded_arguments(np_local2);

  int np_local4 = 42;
  // non-template values are ok still.
  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  np_local4 += p_local0;
}

template <typename T>
void more_template_locals() {
  const T np_local0 = {};
  auto np_local1 = T{};
  T &np_local2 = np_local1;
  T *np_local_ptr = &np_local1;

  const auto np_local3 = T{};
  // FIXME: False positive, the reference points to a template type and needs
  // to be excluded from analysis, but somehow isn't (matchers don't work)
  auto &np_local4 = np_local3;

  const auto *np_local5 = &np_local3;
  auto *np_local6 = &np_local1;

  using TypedefToTemplate = T;
  TypedefToTemplate np_local7{};
  // FIXME: False positive, the reference points to a template type and needs
  // to be excluded from analysis, but somehow isn't (matchers don't work)
  // auto &np_local8 = np_local7;
  const auto &np_local9 = np_local7;
  auto np_local10 = np_local7;
  auto *np_local11 = &np_local10;
  const auto *const np_local12 = &np_local10;

  // FIXME: False positive, the reference points to a template type and needs
  // to be excluded from analysis, but somehow isn't (matchers don't work)
  // TypedefToTemplate &np_local13 = np_local7;
  TypedefToTemplate *np_local14 = &np_local7;
}

void template_instantiation() {
  const int np_local0 = 42;
  int np_local1 = 42;

  define_locals(np_local0, np_local1, np_local0);
  define_locals(np_local1, np_local1, np_local1);
  more_template_locals<int>();
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
  ConstNonConstClass np_local0;

  np_local0.constMethod();
  np_local0.nonConstMethod();

  ConstNonConstClass p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'ConstNonConstClass' can be declared 'const'
  p_local0.constMethod();

  ConstNonConstClass p_local1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'ConstNonConstClass' can be declared 'const'
  double np_local1;
  p_local1.modifyingMethod(np_local1);

  double np_local2;
  ConstNonConstClass p_local2(np_local2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'ConstNonConstClass' can be declared 'const'

  ConstNonConstClass np_local3;
  np_local3.NonConstMember = 42.;

  ConstNonConstClass np_local4;
  np_local4.NonConstMemberRef = 42.;

  ConstNonConstClass p_local3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'ConstNonConstClass' can be declared 'const'
  const double val0 = p_local3.NonConstMember;
  const double val1 = p_local3.NonConstMemberRef;
  const double val2 = *p_local3.NonConstMemberPtr;

  ConstNonConstClass p_local4;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local4' of type 'ConstNonConstClass' can be declared 'const'
  *np_local4.NonConstMemberPtr = 42.;
}

void class_access_array() {
  ConstNonConstClass np_local0[2];
  np_local0[0].constMethod();
  np_local0[1].constMethod();
  np_local0[1].nonConstMethod();

  ConstNonConstClass p_local0[2];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'ConstNonConstClass [2]' can be declared 'const'
  p_local0[0].constMethod();
  np_local0[1].constMethod();
}

struct OperatorsAsConstAsPossible {
  OperatorsAsConstAsPossible &operator+=(const OperatorsAsConstAsPossible &rhs);
  OperatorsAsConstAsPossible operator+(const OperatorsAsConstAsPossible &rhs) const;
};

struct NonConstOperators {
};
NonConstOperators operator+(NonConstOperators &lhs, NonConstOperators &rhs);
NonConstOperators operator-(NonConstOperators lhs, NonConstOperators rhs);

void internal_operator_calls() {
  OperatorsAsConstAsPossible np_local0;
  OperatorsAsConstAsPossible np_local1;
  OperatorsAsConstAsPossible p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'OperatorsAsConstAsPossible' can be declared 'const'
  OperatorsAsConstAsPossible p_local1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'OperatorsAsConstAsPossible' can be declared 'const'

  np_local0 += p_local0;
  np_local1 = p_local0 + p_local1;

  NonConstOperators np_local2;
  NonConstOperators np_local3;
  NonConstOperators np_local4;

  np_local2 = np_local3 + np_local4;

  NonConstOperators p_local2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'NonConstOperators' can be declared 'const'
  NonConstOperators p_local3 = p_local2 - p_local2;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'NonConstOperators' can be declared 'const'
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
  double np_local0[10];
  np_local0[5] = 42.;

  MyVector np_local1;
  np_local1[5] = 42.;

  double p_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double [10]' can be declared 'const'
  double p_local1 = p_local0[5];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'double' can be declared 'const'

  // The following subscript calls suprisingly choose the non-const operator
  // version.
  MyVector np_local2;
  double p_local2 = np_local2[42];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'double' can be declared 'const'

  MyVector np_local3;
  const double np_local4 = np_local3[42];

  // This subscript results in const overloaded operator.
  const MyVector np_local5{};
  double p_local3 = np_local5[42];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'double' can be declared 'const'
}

void const_handle(const double &np_local0);
void const_handle(const double *np_local0);

void non_const_handle(double &np_local0);
void non_const_handle(double *np_local0);

void handle_from_array() {
  // Non-const handle from non-const array forbids declaring the array as const
  double np_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double *p_local0 = &np_local0[1]; // Could be `double *const`, but warning deactivated by default

  double np_local1[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double &non_const_ref = np_local1[1];
  non_const_ref = 42.;

  double np_local2[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double *np_local3;
  np_local3 = &np_local2[5];

  double np_local4[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  non_const_handle(np_local4[2]);
  double np_local5[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  non_const_handle(&np_local5[2]);

  // Constant handles are ok
  double p_local1[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'double [10]' can be declared 'const'
  const double *p_local2 = &p_local1[2]; // Could be `const double *const`, but warning deactivated by default

  double p_local3[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'double [10]' can be declared 'const'
  const double &const_ref = p_local3[2];

  double p_local4[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local4' of type 'double [10]' can be declared 'const'
  const double *const_ptr;
  const_ptr = &p_local4[2];

  double p_local5[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local5' of type 'double [10]' can be declared 'const'
  const_handle(p_local5[2]);
  double p_local6[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local6' of type 'double [10]' can be declared 'const'
  const_handle(&p_local6[2]);
}

void range_for() {
  int np_local0[2] = {1, 2};
  for (int &non_const_ref : np_local0) {
    non_const_ref = 42;
  }

  int np_local1[2] = {1, 2};
  for (auto &non_const_ref : np_local1) {
    non_const_ref = 43;
  }

  int np_local2[2] = {1, 2};
  for (auto &&non_const_ref : np_local2) {
    non_const_ref = 44;
  }

  // FIXME the warning message is suboptimal. It could be defined as
  // `int *const np_local3[2]` because the pointers are not reseated.
  // But this is not easily deducable from the warning.
  int *np_local3[2] = {&np_local0[0], &np_local0[1]};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'np_local3' of type 'int *[2]' can be declared 'const'
  for (int *non_const_ptr : np_local3) {
    *non_const_ptr = 45;
  }

  // FIXME same as above, but silenced
  int *const np_local4[2] = {&np_local0[0], &np_local0[1]};
  for (auto *non_const_ptr : np_local4) {
    *non_const_ptr = 46;
  }

  int p_local0[2] = {1, 2};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int [2]' can be declared 'const'
  for (int value : p_local0) {
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'value' of type 'int' can be declared 'const'
  }

  int p_local1[2] = {1, 2};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int [2]' can be declared 'const'
  for (const int &const_ref : p_local1) {
  }

  int *p_local2[2] = {&np_local0[0], &np_local0[1]};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'int *[2]' can be declared 'const'
  for (const int *con_ptr : p_local2) {
  }

  int *p_local3[2] = {nullptr, nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'int *[2]' can be declared 'const'
  for (const auto *con_ptr : p_local3) {
  }
}

inline void *operator new(decltype(sizeof(void *)), void *p) { return p; }

struct Value {
};
void placement_new() {
  Value Mem;
  Value *V = new (&Mem) Value;
}

struct ModifyingConversion {
  operator int() { return 15; }
};
struct NonModifyingConversion {
  operator int() const { return 15; }
};
void conversion_operators() {
  ModifyingConversion np_local0;
  NonModifyingConversion p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'NonModifyingConversion' can be declared 'const'

  int np_local1 = np_local0;
  np_local1 = p_local0;
}

void casts() {
  decltype(sizeof(void *)) p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'decltype(sizeof(void *))' (aka 'unsigned long') can be declared 'const'
  auto np_local0 = reinterpret_cast<void *>(p_local0);
  np_local0 = nullptr;

  int p_local1 = 43;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'int' can be declared 'const'
  short p_local2 = static_cast<short>(p_local1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'short' can be declared 'const'

  int np_local1 = p_local2;
  int &np_local2 = static_cast<int &>(np_local1);
  np_local2 = 5;
}

void ternary_operator() {
  int np_local0 = 1, np_local1 = 2;
  int &np_local2 = true ? np_local0 : np_local1;
  np_local2 = 2;

  int p_local0 = 3, np_local3 = 5;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  const int &np_local4 = true ? p_local0 : ++np_local3;

  int np_local5[3] = {1, 2, 3};
  int &np_local6 = np_local5[1] < np_local5[2] ? np_local5[0] : np_local5[2];
  np_local6 = 42;

  int np_local7[3] = {1, 2, 3};
  int *np_local8 = np_local7[1] < np_local7[2] ? &np_local7[0] : &np_local7[2];
  *np_local8 = 42;
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
  p_local0.alwaysConst();
  p_local0.sometimesConst();

  TMPClass<double> p_local1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'TMPClass<double>' can be declared 'const'
  p_local1.alwaysConst();

  TMPClass<double> np_local0;
  np_local0.alwaysConst();
  np_local0.sometimesConst();
}

// This test is the essence from llvm/lib/Support/MemoryBuffer.cpp at line 450
template <typename T>
struct to_construct : T {
  to_construct(int &j) {}
};
template <typename T>
void placement_new_in_unique_ptr() {
  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  int np_local0 = p_local0;
  new to_construct<T>(np_local0);
}

struct stream_obj {};
stream_obj &operator>>(stream_obj &o, unsigned &foo);
void input_operator() {
  stream_obj np_local0;
  unsigned np_local1 = 42;
  np_local0 >> np_local1;
}

struct stream_obj_template {};
template <typename IStream>
IStream &operator>>(IStream &o, unsigned &foo);

template <typename Stream>
void input_operator_template() {
  Stream np_local0;
  unsigned np_local1 = 42;
  np_local0 >> np_local1;
}

// Test bit fields
struct HardwareRegister {
  unsigned field : 5;
  unsigned : 7;
  unsigned another : 20;
};

void TestRegisters() {
  HardwareRegister np_reg0;
  np_reg0.field = 3;

  HardwareRegister p_reg1{3, 22};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_reg1' of type 'HardwareRegister' can be declared 'const'
  const unsigned p_val = p_reg1.another;
}

struct IntWrapper {
  IntWrapper &operator=(unsigned value) { return *this; }
  template <typename Istream>
  friend Istream &operator>>(Istream &is, IntWrapper &rhs);
};
struct IntMaker {
  friend IntMaker &operator>>(IntMaker &, unsigned &);
};
template <typename Istream>
Istream &operator>>(Istream &is, IntWrapper &rhs) {
  unsigned np_local0 = 0;
  is >> np_local0;
  return is;
}

struct Actuator {
  int actuations;
};
struct Sensor {
  int observations;
};
struct System : public Actuator, public Sensor {
};
int some_computation(int arg);
int test_inheritance() {
  System np_sys;
  np_sys.actuations = 5;
  return some_computation(np_sys.actuations);
}
struct AnotherActuator : Actuator {
};
Actuator &test_return_polymorphic() {
  static AnotherActuator np_local0;
  return np_local0;
}

using f_signature = int *(*)(int &);
int *my_alloc(int &size) { return new int[size]; }
struct A {
  int f(int &i) { return i + 1; }
  int (A::*x)(int &);
};
void f() {
  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  int np_local0 = 42;
  f_signature action = my_alloc;
  action(np_local0);
  my_alloc(np_local0);

  int np_local1 = 42;
  A a;
  a.x = &A::f;
  (a.*(a.x))(np_local1);
}

struct nested_data {
  int more_data;
};
struct repro_assignment_to_reference {
  int my_data;
  nested_data nested;
};
void assignment_reference() {
  repro_assignment_to_reference np_local0{42};
  int &np_local1 = np_local0.my_data;
  np_local1++;

  repro_assignment_to_reference np_local2;
  int &np_local3 = np_local2.nested.more_data;
  np_local3++;
}

struct non_const_iterator {
  int data[42];

  int *begin() { return &data[0]; }
  int *end() { return &data[41]; }
};

// The problem is, that 'begin()' and 'end()' are not const overloaded, so
// they are always a mutation. If 'np_local1' is fixed to const it results in
// a compilation error.
void for_bad_iterators() {
  non_const_iterator np_local0;
  non_const_iterator &np_local1 = np_local0;

  for (int np_local2 : np_local1) {
    np_local2++;
  }

  non_const_iterator np_local3;
  for (int p_local0 : np_local3)
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'p_local0' of type 'int' can be declared 'const'
    ;

  // Horrible code constructs...
  {
    non_const_iterator np_local4;
    np_local4.data[0]++;
    non_const_iterator np_local5;
    for (int p_local1 : np_local4, np_local5)
      // CHECK-MESSAGES: [[@LINE-1]]:10: warning: variable 'p_local1' of type 'int' can be declared 'const'
      ;

    non_const_iterator np_local6;
    non_const_iterator np_local7;
    for (int p_local2 : 1 > 2 ? np_local6 : np_local7)
      // CHECK-MESSAGES: [[@LINE-1]]:10: warning: variable 'p_local2' of type 'int' can be declared 'const'
      ;

    non_const_iterator np_local8;
    non_const_iterator np_local9;
    for (int p_local3 : 2 > 1 ? np_local8 : (np_local8, np_local9))
      // CHECK-MESSAGES: [[@LINE-1]]:10: warning: variable 'p_local3' of type 'int' can be declared 'const'
      ;
  }
}

struct good_iterator {
  int data[3] = {1, 2, 3};

  int *begin() { return &data[0]; }
  int *end() { return &data[2]; }
  const int *begin() const { return &data[0]; }
  const int *end() const { return &data[2]; }
};

void good_iterators() {
  good_iterator p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'good_iterator' can be declared 'const'
  good_iterator &p_local1 = p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'good_iterator &' can be declared 'const'

  for (int p_local2 : p_local1) {
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'p_local2' of type 'int' can be declared 'const'
    (void)p_local2;
  }

  good_iterator p_local3;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local3' of type 'good_iterator' can be declared 'const'
  for (int p_local4 : p_local3)
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'p_local4' of type 'int' can be declared 'const'
    ;
  good_iterator np_local1;
  for (int &np_local2 : np_local1)
    np_local2++;
}

void for_bad_iterators_array() {
  int np_local0[42];
  int(&np_local1)[42] = np_local0;

  for (int &np_local2 : np_local1) {
    np_local2++;
  }
}
void for_ok_iterators_array() {
  int np_local0[42];
  int(&p_local0)[42] = np_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int (&)[42]' can be declared 'const'

  for (int p_local1 : p_local0) {
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'p_local1' of type 'int' can be declared 'const'
    (void)p_local1;
  }
}

void take_ref(int &);
void ternary_reference() {
  int np_local0 = 42;
  int np_local1 = 43;
  take_ref((np_local0 > np_local1 ? np_local0 : (np_local0, np_local1)));
}

void complex_usage() {
  int np_local0 = 42;
  int p_local0 = 42;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int' can be declared 'const'
  int np_local1 = 42;
  (np_local0 == p_local0 ? np_local0 : (p_local0, np_local1))++;
}

template <typename T>
struct SmallVectorBase {
  T data[4];
  void push_back(const T &el) {}
  int size() const { return 4; }
  T *begin() { return data; }
  const T *begin() const { return data; }
  T *end() { return data + 4; }
  const T *end() const { return data + 4; }
};

template <typename T>
struct SmallVector : SmallVectorBase<T> {};

template <class T>
void EmitProtocolMethodList(T &&Methods) {
  // Note: If the template is uninstantiated the analysis does not figure out,
  // that p_local0 could be const. Not sure why, but probably bails because
  // some expressions are type-dependant.
  SmallVector<const int *> p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'SmallVector<const int *>' can be declared 'const'
  SmallVector<const int *> np_local0;
  for (const auto *I : Methods) {
    if (I == nullptr)
      np_local0.push_back(I);
  }
  p_local0.size();
}
void instantiate() {
  int *p_local0[4] = {nullptr, nullptr, nullptr, nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int *[4]' can be declared 'const'
  EmitProtocolMethodList(p_local0);
}
struct base {
  int member;
};
struct derived : base {};
struct another_struct {
  derived member;
};
void another_struct_f() {
  another_struct np_local0{};
  base &np_local1 = np_local0.member;
  np_local1.member++;
}
struct list_init {
  int &member;
};
void create_false_positive() {
  int np_local0 = 42;
  list_init p_local0 = {np_local0};
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'p_local0' of type 'list_init' can be declared 'const'
}
struct list_init_derived {
  base &member;
};
void list_init_derived_func() {
  derived np_local0;
  list_init_derived p_local0 = {np_local0};
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'p_local0' of type 'list_init_derived' can be declared 'const'
}
template <typename L, typename R>
struct ref_pair {
  L &first;
  R &second;
};
template <typename T>
void list_init_template() {
  T np_local0{};
  ref_pair<T, T> p_local0 = {np_local0, np_local0};
}
void cast_in_class_hierarchy() {
  derived np_local0;
  base p_local1 = static_cast<base &>(np_local0);
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'p_local1' of type 'base' can be declared 'const'
}

void function_ref_target(int);
using my_function_type = void (&)(int);
void func_references() {
  // Could be const, because the reference is not adjusted but adding that
  // has no effect and creates a compiler warning.
  my_function_type ptr = function_ref_target;
}

#if 0
template <typename T>
T &return_ref() {
  static T global;
  return global;
}
template <typename T>
T *return_ptr() { return &return_ref<T>(); }

template <typename T>
void auto_usage_variants() {
  // FIXME: Currently all 'auto's that deduce to a reference are not ignored
  // for the analysis. That results in bad transformations.
  auto auto_val0 = T{};
  auto &auto_val1 = auto_val0; // Bad
  auto *auto_val2 = &auto_val0;

  auto auto_ref0 = return_ref<T>();  // Bad
  auto &auto_ref1 = return_ref<T>(); // Bad
  auto *auto_ref2 = return_ptr<T>();

  auto auto_ptr0 = return_ptr<T>();
  auto &auto_ptr1 = auto_ptr0;
  auto *auto_ptr2 = return_ptr<T>();

  using MyTypedef = T;
  auto auto_td0 = MyTypedef{};
  auto &auto_td1 = auto_td0; // Bad
  auto *auto_td2 = &auto_td0;
}
void instantiate_auto_cases() {
  auto_usage_variants<int>();
  auto_usage_variants<System>();
}
#endif
