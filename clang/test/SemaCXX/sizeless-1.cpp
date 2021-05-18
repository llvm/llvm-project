// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify -W -Wall -Wno-unused-but-set-variable -Wrange-loop-analysis -triple arm64-linux-gnu -target-feature +sve -std=c++98 %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify -W -Wall -Wno-unused-but-set-variable -Wrange-loop-analysis -triple arm64-linux-gnu -target-feature +sve -std=c++11 %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify -W -Wall -Wno-unused-but-set-variable -Wrange-loop-analysis -triple arm64-linux-gnu -target-feature +sve -std=c++17 %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify -W -Wall -Wno-unused-but-set-variable -Wrange-loop-analysis -triple arm64-linux-gnu -target-feature +sve -std=gnu++17 %s

namespace std {
struct type_info;
}

typedef __SVInt8_t svint8_t;
typedef __SVInt16_t svint16_t;

svint8_t global_int8;          // expected-error {{non-local variable with sizeless type 'svint8_t'}}
extern svint8_t extern_int8;   // expected-error {{non-local variable with sizeless type 'svint8_t'}}
static svint8_t static_int8;   // expected-error {{non-local variable with sizeless type 'svint8_t'}}
__thread svint8_t thread_int8; // expected-error {{non-local variable with sizeless type 'svint8_t'}}
svint8_t *global_int8_ptr;
extern svint8_t *extern_int8_ptr;
static svint8_t *static_int8_ptr;

typedef svint8_t int8_typedef;
typedef svint8_t *int8_ptr_typedef;

int sizeof_int8 = sizeof(svint8_t);             // expected-error {{invalid application of 'sizeof' to sizeless type 'svint8_t'}}
int sizeof_int8_var = sizeof(*extern_int8_ptr); // expected-error {{invalid application of 'sizeof' to sizeless type 'svint8_t'}}
int sizeof_int8_var_ptr = sizeof(extern_int8_ptr);

#if __cplusplus >= 201103L
int alignof_int8 = alignof(svint8_t);                // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}}
int alignof_int8_var = alignof(*extern_int8_ptr);    // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}} expected-warning {{GNU extension}}
int alignof_int8_var_ptr = alignof(extern_int8_ptr); // expected-warning {{GNU extension}}
#else
int alignof_int8 = _Alignof(svint8_t);                // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}}
int alignof_int8_var = _Alignof(*extern_int8_ptr);    // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}} expected-warning {{GNU extension}}
int alignof_int8_var_ptr = _Alignof(extern_int8_ptr); // expected-warning {{GNU extension}}
#endif

void pass_int8(svint8_t); // expected-note {{no known conversion}}

svint8_t return_int8();

typedef svint8_t vec_int8_a __attribute__((vector_size(64)));    // expected-error {{invalid vector element type}}
typedef svint8_t vec_int8_b __attribute__((ext_vector_type(4))); // expected-error {{invalid vector element type}}

void dump(const volatile void *);

void overf(svint8_t);
void overf(svint16_t);

void overf8(svint8_t); // expected-note + {{not viable}}
void overf8(int);      // expected-note + {{not viable}}

void overf16(svint16_t); // expected-note + {{not viable}}
void overf16(int);       // expected-note + {{not viable}}

void varargs(int, ...);

void unused() {
  svint8_t unused_int8; // expected-warning {{unused}}
}

struct incomplete_struct *incomplete_ptr;

typedef svint8_t sizeless_array[1]; // expected-error {{array has sizeless element type}}

void func(int sel) {
  static svint8_t static_int8; // expected-error {{non-local variable with sizeless type 'svint8_t'}}

  svint8_t local_int8;
  svint16_t local_int16;

  svint8_t __attribute__((aligned)) aligned_int8_1;    // expected-error {{'aligned' attribute cannot be applied to sizeless type 'svint8_t'}}
  svint8_t __attribute__((aligned(4))) aligned_int8_2; // expected-error {{'aligned' attribute cannot be applied to sizeless type 'svint8_t'}}
  svint8_t _Alignas(int) aligned_int8_3;               // expected-error {{'_Alignas' attribute cannot be applied to sizeless type 'svint8_t'}}

  int _Alignas(svint8_t) aligned_int; // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}}

  // Using pointers to sizeless data isn't wrong here, but because the
  // type is incomplete, it doesn't provide any alignment guarantees.
  _Static_assert(__atomic_is_lock_free(1, &local_int8) == __atomic_is_lock_free(1, incomplete_ptr), "");
  _Static_assert(__atomic_is_lock_free(2, &local_int8) == __atomic_is_lock_free(2, incomplete_ptr), ""); // expected-error {{static_assert expression is not an integral constant expression}}
  _Static_assert(__atomic_always_lock_free(1, &local_int8) == __atomic_always_lock_free(1, incomplete_ptr), "");

  local_int8; // expected-warning {{expression result unused}}

  (void)local_int8;

  local_int8, 0; // expected-warning + {{expression result unused}}

  0, local_int8; // expected-warning + {{expression result unused}}

  svint8_t init_int8 = local_int8;
  svint8_t bad_init_int8 = for; // expected-error {{expected expression}}

#if __cplusplus >= 201103L
  int empty_brace_init_int = {};
  svint8_t empty_brace_init_int8 = {};
#else
  int empty_brace_init_int = {}; // expected-error {{scalar initializer cannot be empty}}
  svint8_t empty_brace_init_int8 = {}; // expected-error {{initializer for sizeless type 'svint8_t' (aka '__SVInt8_t') cannot be empty}}
#endif
  svint8_t brace_init_int8 = {local_int8};
  svint8_t bad_brace_init_int8_1 = {local_int8, 0};    // expected-error {{excess elements in initializer for indivisible sizeless type 'svint8_t'}}
  svint8_t bad_brace_init_int8_2 = {0};                // expected-error {{rvalue of type 'int'}}
  svint8_t bad_brace_init_int8_3 = {local_int16};      // expected-error {{lvalue of type 'svint16_t'}}
  svint8_t bad_brace_init_int8_4 = {[0] = local_int8}; // expected-error {{designator in initializer for indivisible sizeless type 'svint8_t'}} expected-warning {{array designators are a C99 extension}}
  svint8_t bad_brace_init_int8_5 = {{local_int8}};     // expected-warning {{too many braces around initializer}}
  svint8_t bad_brace_init_int8_6 = {{local_int8, 0}};  // expected-warning {{too many braces around initializer}}

  const svint8_t const_int8 = local_int8; // expected-note {{declared const here}}
  const svint8_t uninit_const_int8;       // expected-error {{default initialization of an object of const type 'const svint8_t'}}

  volatile svint8_t volatile_int8;

  const volatile svint8_t const_volatile_int8 = local_int8; // expected-note {{declared const here}}
  const volatile svint8_t uninit_const_volatile_int8;       // expected-error {{default initialization of an object of const type 'const volatile svint8_t'}}

  _Atomic svint8_t atomic_int8;      // expected-error {{_Atomic cannot be applied to sizeless type 'svint8_t'}}
  __restrict svint8_t restrict_int8; // expected-error {{requires a pointer or reference}}

  svint8_t array_int8[1];          // expected-error {{array has sizeless element type}}
  svint8_t array_int8_init[] = {}; // expected-error {{array has sizeless element type}}

  bool test_int8 = init_int8; // expected-error {{cannot initialize a variable of type 'bool' with an lvalue of type 'svint8_t'}}

  int int_int8 = init_int8; // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type 'svint8_t'}}

  init_int8 = local_int8;
  init_int8 = local_int16; // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'svint16_t'}}
  init_int8 = sel;         // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'int'}}

  sel = local_int8; // expected-error {{assigning to 'int' from incompatible type 'svint8_t'}}

  local_int8 = (svint8_t)local_int8;
  local_int8 = (const svint8_t)local_int8;
  local_int8 = (svint8_t)local_int16; // expected-error {{C-style cast from 'svint16_t' (aka '__SVInt16_t') to 'svint8_t' (aka '__SVInt8_t') is not allowed}}
  local_int8 = (svint8_t)0;           // expected-error {{C-style cast from 'int' to 'svint8_t' (aka '__SVInt8_t') is not allowed}}
  sel = (int)local_int8;              // expected-error {{C-style cast from 'svint8_t' (aka '__SVInt8_t') to 'int' is not allowed}}

  init_int8 = local_int8;
  init_int8 = const_int8;
  init_int8 = volatile_int8;
  init_int8 = const_volatile_int8;

  const_int8 = local_int8; // expected-error {{cannot assign to variable 'const_int8' with const-qualified type 'const svint8_t'}}

  volatile_int8 = local_int8;
  volatile_int8 = const_int8;
  volatile_int8 = volatile_int8;
  volatile_int8 = const_volatile_int8;

  const_volatile_int8 = local_int8; // expected-error {{cannot assign to variable 'const_volatile_int8' with const-qualified type 'const volatile svint8_t'}}

  init_int8 = sel ? init_int8 : local_int8;
  init_int8 = sel ? init_int8 : const_int8;
  init_int8 = sel ? volatile_int8 : const_int8;
  init_int8 = sel ? volatile_int8 : const_volatile_int8;

  pass_int8(local_int8);
  pass_int8(local_int16); // expected-error {{no matching function}}

  local_int8 = return_int8();
  local_int16 = return_int8(); // expected-error {{assigning to 'svint16_t' (aka '__SVInt16_t') from incompatible type 'svint8_t'}}

  dump(&local_int8);
  dump(&const_int8);
  dump(&volatile_int8);
  dump(&const_volatile_int8);

  dump(&local_int8 + 1); // expected-error {{arithmetic on a pointer to sizeless type}}

  *&local_int8 = local_int8;
  *&const_int8 = local_int8; // expected-error {{read-only variable is not assignable}}
  *&volatile_int8 = local_int8;
  *&const_volatile_int8 = local_int8; // expected-error {{read-only variable is not assignable}}

  global_int8_ptr[0] = local_int8;       // expected-error {{subscript of pointer to sizeless type 'svint8_t'}}
  global_int8_ptr[1] = local_int8;       // expected-error {{subscript of pointer to sizeless type 'svint8_t'}}
  global_int8_ptr = &global_int8_ptr[2]; // expected-error {{subscript of pointer to sizeless type 'svint8_t'}}

  overf(local_int8);
  overf(local_int16);

  overf8(local_int8);
  overf8(local_int16); // expected-error {{no matching function}}

  overf16(local_int8); // expected-error {{no matching function}}
  overf16(local_int16);

  varargs(1, local_int8, local_int16);

  global_int8_ptr++;                 // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr--;                 // expected-error {{arithmetic on a pointer to sizeless type}}
  ++global_int8_ptr;                 // expected-error {{arithmetic on a pointer to sizeless type}}
  --global_int8_ptr;                 // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr + 1;               // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr - 1;               // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr += 1;              // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr -= 1;              // expected-error {{arithmetic on a pointer to sizeless type}}
  global_int8_ptr - global_int8_ptr; // expected-error {{arithmetic on a pointer to sizeless type}}

  +init_int8;       // expected-error {{invalid argument type 'svint8_t'}}
  ++init_int8;      // expected-error {{cannot increment value of type 'svint8_t'}}
  init_int8++;      // expected-error {{cannot increment value of type 'svint8_t'}}
  -init_int8;       // expected-error {{invalid argument type 'svint8_t'}}
  --init_int8;      // expected-error {{cannot decrement value of type 'svint8_t'}}
  init_int8--;      // expected-error {{cannot decrement value of type 'svint8_t'}}
  ~init_int8;       // expected-error {{invalid argument type 'svint8_t'}}
  !init_int8;       // expected-error {{invalid argument type 'svint8_t'}}
  *init_int8;       // expected-error {{indirection requires pointer operand}}
  __real init_int8; // expected-error {{invalid type 'svint8_t'}}
  __imag init_int8; // expected-error {{invalid type 'svint8_t'}}

  local_int8 + init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 - init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 *init_int8;   // expected-error {{invalid operands to binary expression}}
  local_int8 / init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 % init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 &init_int8;   // expected-error {{invalid operands to binary expression}}
  local_int8 | init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 ^ init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 << init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 >> init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 < init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 <= init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 == init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 != init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 >= init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 > init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 &&init_int8;  // expected-error {{invalid operands to binary expression}} expected-error {{not contextually convertible}}
  local_int8 || init_int8; // expected-error {{invalid operands to binary expression}} expected-error {{not contextually convertible}}

  local_int8 += init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 -= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 *= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 /= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 %= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 &= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 |= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 ^= init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 <<= init_int8; // expected-error {{invalid operands to binary expression}}
  local_int8 >>= init_int8; // expected-error {{invalid operands to binary expression}}

  local_int8 + 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 - 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 * 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 / 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 % 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 & 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 | 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 ^ 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 << 0; // expected-error {{invalid operands to binary expression}}
  local_int8 >> 0; // expected-error {{invalid operands to binary expression}}
  local_int8 < 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 <= 0; // expected-error {{invalid operands to binary expression}}
  local_int8 == 0; // expected-error {{invalid operands to binary expression}}
  local_int8 != 0; // expected-error {{invalid operands to binary expression}}
  local_int8 >= 0; // expected-error {{invalid operands to binary expression}}
  local_int8 > 0;  // expected-error {{invalid operands to binary expression}}
  local_int8 && 0; // expected-error {{invalid operands to binary expression}} expected-error {{not contextually convertible}}
  local_int8 || 0; // expected-error {{invalid operands to binary expression}} expected-error {{not contextually convertible}}

  if (local_int8) { // expected-error {{not contextually convertible to 'bool'}}
  }
  while (local_int8) { // expected-error {{not contextually convertible to 'bool'}}
  }
  do {
  } while (local_int8); // expected-error {{not contextually convertible to 'bool'}}
  switch (local_int8) { // expected-error {{statement requires expression of integer type}}
  default:;
  }
}

int vararg_receiver(int count, svint8_t first, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, first);
  __builtin_va_arg(va, svint8_t);
  __builtin_va_end(va);
  return count;
}

struct sized_struct {
  int f1;
  svint8_t f2;     // expected-error {{field has sizeless type 'svint8_t'}}
  svint8_t f3 : 2; // expected-error {{field has sizeless type 'svint8_t'}}
  svint8_t : 3;    // expected-error {{field has sizeless type 'svint8_t'}}
};

union sized_union {
  int f1;
  svint8_t f2;     // expected-error {{field has sizeless type 'svint8_t'}}
  svint8_t f3 : 2; // expected-error {{field has sizeless type 'svint8_t'}}
  svint8_t : 3;    // expected-error {{field has sizeless type 'svint8_t'}}
};

void pass_int8_ref(svint8_t &); // expected-note {{not viable}}

svint8_t &return_int8_ref();
#if __cplusplus >= 201103L
svint8_t &&return_int8_rvalue_ref();
#endif

template <typename T>
struct s_template {
  T y; // expected-error {{field has sizeless type '__SVInt8_t'}}
};

template <typename T>
struct s_ptr_template {
  s_ptr_template();
  s_ptr_template(T, svint8_t = svint8_t());
  s_ptr_template(const s_ptr_template &, svint8_t = svint8_t());
  T *y;
};

template <typename T>
struct s_array_template {
  T y[1]; // expected-error {{array has sizeless element type}}
};

struct widget {
  widget(s_ptr_template<int>);
  svint8_t operator[](int);
};

template <typename T>
struct wrapper_iterator {
  T operator++();
  T operator*() const;
  bool operator!=(const wrapper_iterator &) const;
};

template <typename T>
struct wrapper {
  wrapper();
  operator T() const;
  wrapper_iterator<T> begin() const;
  wrapper_iterator<T> end() const;
};

#if __cplusplus >= 201103L
struct explicit_conv {
  explicit operator svint8_t() const; // expected-note {{explicit conversion function is not a candidate}}
};
#endif

struct constructible_from_sizeless {
  constructible_from_sizeless(svint8_t);
};

void with_default(svint8_t = svint8_t());

#if __cplusplus >= 201103L
constexpr int ce_taking_int8(svint8_t) { return 1; } // expected-error {{constexpr function's 1st parameter type 'svint8_t' (aka '__SVInt8_t') is not a literal type}}
#endif

#if __cplusplus < 201703L
void throwing_func() throw(svint8_t); // expected-error {{sizeless type 'svint8_t' (aka '__SVInt8_t') is not allowed in exception specification}}
void throwing_pointer_func() throw(svint8_t *);
void throwing_reference_func() throw(svint8_t &); // expected-error {{reference to sizeless type 'svint8_t' (aka '__SVInt8_t') is not allowed in exception specification}}
#endif

template <typename T>
void template_fn_direct(T) {}
template <typename T>
void template_fn_ref(T &) {}
template <typename T>
void template_fn_const_ref(const T &) {}
#if __cplusplus >= 201103L
template <typename T>
void template_fn_rvalue_ref(T &&) {}
#endif

#if __cplusplus >= 201103L
template <typename T>
using array_alias = T[1]; // expected-error {{array has sizeless element type '__SVInt8_t'}}
extern array_alias<int> *array_alias_int_ptr;
extern array_alias<svint8_t> *array_alias_int8_ptr; // expected-note {{in instantiation of template type alias 'array_alias' requested here}}
#endif

extern "C" svint8_t c_return_int8();

void cxx_only(int sel) {
  svint8_t local_int8;
  svint16_t local_int16;

  pass_int8_ref(local_int8);
  pass_int8_ref(local_int16); // expected-error {{no matching function}}

  local_int8 = return_int8_ref();
  local_int16 = return_int8_ref(); // expected-error {{assigning to 'svint16_t' (aka '__SVInt16_t') from incompatible type 'svint8_t'}}
  return_int8_ref() = local_int8;
  return_int8_ref() = local_int16; // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'svint16_t'}}

#if __cplusplus >= 201103L
  local_int8 = return_int8_rvalue_ref();
  local_int16 = return_int8_rvalue_ref(); // expected-error {{assigning to 'svint16_t' (aka '__SVInt16_t') from incompatible type 'svint8_t'}}

  return_int8_rvalue_ref() = local_int8;  // expected-error {{expression is not assignable}}
  return_int8_rvalue_ref() = local_int16; // expected-error {{expression is not assignable}}
#endif

  local_int8 = static_cast<svint8_t>(local_int8);
  local_int8 = static_cast<svint8_t>(local_int16);  // expected-error {{static_cast from 'svint16_t' (aka '__SVInt16_t') to 'svint8_t' (aka '__SVInt8_t') is not allowed}}
  local_int8 = static_cast<svint8_t>(0);            // expected-error {{static_cast from 'int' to 'svint8_t' (aka '__SVInt8_t') is not allowed}}
  local_int16 = static_cast<svint16_t>(local_int8); // expected-error {{static_cast from 'svint8_t' (aka '__SVInt8_t') to 'svint16_t' (aka '__SVInt16_t') is not allowed}}
  sel = static_cast<int>(local_int8);               // expected-error {{static_cast from 'svint8_t' (aka '__SVInt8_t') to 'int' is not allowed}}

  throw local_int8; // expected-error {{cannot throw object of sizeless type 'svint8_t'}}
  throw global_int8_ptr;

  try {
  } catch (int) {
  }
  try {
  } catch (svint8_t) { // expected-error {{cannot catch sizeless type 'svint8_t'}}
  }
  try {
  } catch (svint8_t *) {
  }
  try {
  } catch (svint8_t &) { // expected-error {{cannot catch reference to sizeless type 'svint8_t'}}
  }

  new svint8_t;     // expected-error {{allocation of sizeless type 'svint8_t'}}
  new svint8_t();   // expected-error {{allocation of sizeless type 'svint8_t'}}
  new svint8_t[10]; // expected-error {{allocation of sizeless type 'svint8_t'}}
  new svint8_t *;

  new (global_int8_ptr) svint8_t;     // expected-error {{allocation of sizeless type 'svint8_t'}}
  new (global_int8_ptr) svint8_t();   // expected-error {{allocation of sizeless type 'svint8_t'}}
  new (global_int8_ptr) svint8_t[10]; // expected-error {{allocation of sizeless type 'svint8_t'}}

  delete global_int8_ptr;   // expected-error {{cannot delete expression of type 'svint8_t *'}}
  delete[] global_int8_ptr; // expected-error {{cannot delete expression of type 'svint8_t *'}}

  local_int8.~__SVInt8_t(); // expected-error {{object expression of non-scalar type 'svint8_t' (aka '__SVInt8_t') cannot be used in a pseudo-destructor expression}}

  (void)svint8_t();

  local_int8 = svint8_t();
  local_int8 = svint16_t(); // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'svint16_t'}}

  s_template<int> st_int;
  s_template<svint8_t> st_svint8; // expected-note {{in instantiation}}

  s_ptr_template<int> st_ptr_int;
  s_ptr_template<svint8_t> st_ptr_svint8;

  widget w(1);
  local_int8 = w[1];

  s_array_template<int> st_array_int;
  s_array_template<svint8_t> st_array_svint8; // expected-note {{in instantiation}}

  local_int8 = static_cast<svint8_t>(wrapper<svint8_t>());
  local_int16 = static_cast<svint8_t>(wrapper<svint8_t>()); // expected-error {{assigning to 'svint16_t' (aka '__SVInt16_t') from incompatible type 'svint8_t'}}

  local_int8 = wrapper<svint8_t>();
  local_int16 = wrapper<svint8_t>(); // expected-error {{assigning to 'svint16_t' (aka '__SVInt16_t') from incompatible type 'wrapper<svint8_t>'}}

  svint8_t &ref_int8 = local_int8;
  ref_int8 = ref_int8; // expected-warning {{explicitly assigning value of variable of type 'svint8_t' (aka '__SVInt8_t') to itself}}
  ref_int8 = local_int8;
  local_int8 = ref_int8;

#if __cplusplus >= 201103L
  svint8_t zero_init_int8{};
  svint8_t init_int8{local_int8};
  svint8_t bad_brace_init_int8_1{local_int8, 0};    // expected-error {{excess elements in initializer for indivisible sizeless type 'svint8_t'}}
  svint8_t bad_brace_init_int8_2{0};                // expected-error {{rvalue of type 'int'}}
  svint8_t bad_brace_init_int8_3{local_int16};      // expected-error {{lvalue of type 'svint16_t'}}
  svint8_t bad_brace_init_int8_4{[0] = local_int8}; // expected-error {{designator in initializer for indivisible sizeless type 'svint8_t'}} expected-warning {{array designators are a C99 extension}}
  svint8_t bad_brace_init_int8_5{{local_int8}};     // expected-warning {{too many braces around initializer}}
  svint8_t bad_brace_init_int8_6{{local_int8, 0}};  // expected-warning {{too many braces around initializer}}
  svint8_t wrapper_init_int8{wrapper<svint8_t>()};
  svint8_t &ref_init_int8{local_int8};

  template_fn_direct<svint8_t>({wrapper<svint8_t>()});
#endif

  template_fn_direct(local_int8);
  template_fn_ref(local_int8);
  template_fn_const_ref(local_int8);
#if __cplusplus >= 201103L
  template_fn_rvalue_ref(local_int8);
#endif

#if __cplusplus >= 201103L
  constexpr svint8_t ce_int8_a = wrapper<svint8_t>(); // expected-error {{constexpr variable cannot have non-literal type 'const svint8_t'}}
#endif

  (void)typeid(__SVInt8_t);
  (void)typeid(__SVInt8_t *);
  (void)typeid(local_int8);
  (void)typeid(ref_int8);
  (void)typeid(static_int8_ptr);

  _Static_assert(__is_trivially_copyable(svint8_t), "");
  _Static_assert(__is_trivially_destructible(svint8_t), "");
  _Static_assert(!__is_nothrow_assignable(svint8_t, svint8_t), "");
  _Static_assert(__is_nothrow_assignable(svint8_t &, svint8_t), "");
  _Static_assert(!__is_nothrow_assignable(svint8_t &, svint16_t), "");
  _Static_assert(__is_constructible(svint8_t), "");
  _Static_assert(__is_constructible(svint8_t, svint8_t), "");
  _Static_assert(!__is_constructible(svint8_t, svint8_t, svint8_t), "");
  _Static_assert(!__is_constructible(svint8_t, svint16_t), "");
  _Static_assert(__is_nothrow_constructible(svint8_t), "");
  _Static_assert(__is_nothrow_constructible(svint8_t, svint8_t), "");
  _Static_assert(!__is_nothrow_constructible(svint8_t, svint16_t), "");
  _Static_assert(!__is_assignable(svint8_t, svint8_t), "");
  _Static_assert(__is_assignable(svint8_t &, svint8_t), "");
  _Static_assert(!__is_assignable(svint8_t &, svint16_t), "");
  _Static_assert(__has_nothrow_assign(svint8_t), "");
  _Static_assert(__has_nothrow_move_assign(svint8_t), "");
  _Static_assert(__has_nothrow_copy(svint8_t), "");
  _Static_assert(__has_nothrow_constructor(svint8_t), "");
  _Static_assert(__has_trivial_assign(svint8_t), "");
  _Static_assert(__has_trivial_move_assign(svint8_t), "");
  _Static_assert(__has_trivial_copy(svint8_t), "");
  _Static_assert(__has_trivial_constructor(svint8_t), "");
  _Static_assert(__has_trivial_move_constructor(svint8_t), "");
  _Static_assert(__has_trivial_destructor(svint8_t), "");
  _Static_assert(!__has_virtual_destructor(svint8_t), "");
  _Static_assert(!__is_abstract(svint8_t), "");
  _Static_assert(!__is_aggregate(svint8_t), "");
  _Static_assert(!__is_base_of(svint8_t, svint8_t), "");
  _Static_assert(!__is_class(svint8_t), "");
  _Static_assert(__is_convertible_to(svint8_t, svint8_t), "");
  _Static_assert(!__is_convertible_to(svint8_t, svint16_t), "");
  _Static_assert(!__is_empty(svint8_t), "");
  _Static_assert(!__is_enum(svint8_t), "");
  _Static_assert(!__is_final(svint8_t), "");
  _Static_assert(!__is_literal(svint8_t), "");
  _Static_assert(__is_pod(svint8_t), "");
  _Static_assert(!__is_polymorphic(svint8_t), "");
  _Static_assert(__is_trivial(svint8_t), "");
  _Static_assert(__is_object(svint8_t), "");
  _Static_assert(!__is_arithmetic(svint8_t), "");
  _Static_assert(!__is_floating_point(svint8_t), "");
  _Static_assert(!__is_integral(svint8_t), "");
  _Static_assert(!__is_void(svint8_t), "");
  _Static_assert(!__is_array(svint8_t), "");
  _Static_assert(!__is_function(svint8_t), "");
  _Static_assert(!__is_reference(svint8_t), "");
  _Static_assert(__is_reference(svint8_t &), "");
  _Static_assert(__is_reference(const svint8_t &), "");
  _Static_assert(!__is_lvalue_reference(svint8_t), "");
  _Static_assert(__is_lvalue_reference(svint8_t &), "");
#if __cplusplus >= 201103L
  _Static_assert(!__is_lvalue_reference(svint8_t &&), "");
#endif
  _Static_assert(!__is_rvalue_reference(svint8_t), "");
  _Static_assert(!__is_rvalue_reference(svint8_t &), "");
#if __cplusplus >= 201103L
  _Static_assert(__is_rvalue_reference(svint8_t &&), "");
#endif
  _Static_assert(!__is_fundamental(svint8_t), "");
  _Static_assert(__is_object(svint8_t), "");
  _Static_assert(!__is_scalar(svint8_t), "");
  _Static_assert(!__is_compound(svint8_t), "");
  _Static_assert(!__is_pointer(svint8_t), "");
  _Static_assert(__is_pointer(svint8_t *), "");
  _Static_assert(!__is_member_object_pointer(svint8_t), "");
  _Static_assert(!__is_member_function_pointer(svint8_t), "");
  _Static_assert(!__is_member_pointer(svint8_t), "");
  _Static_assert(!__is_const(svint8_t), "");
  _Static_assert(__is_const(const svint8_t), "");
  _Static_assert(__is_const(const volatile svint8_t), "");
  _Static_assert(!__is_volatile(svint8_t), "");
  _Static_assert(__is_volatile(volatile svint8_t), "");
  _Static_assert(__is_volatile(const volatile svint8_t), "");
  _Static_assert(!__is_standard_layout(svint8_t), "");
  // At present these types are opaque and don't have the properties
  // implied by their name.
  _Static_assert(!__is_signed(svint8_t), "");
  _Static_assert(!__is_unsigned(svint8_t), "");

#if __cplusplus >= 201103L
  auto auto_int8 = local_int8;
  auto auto_int16 = local_int16;
#if __cplusplus >= 201703L
  auto [auto_int8_a] = local_int8; // expected-error {{cannot decompose non-class, non-array type '__SVInt8_t'}}
#endif
#endif

  s_ptr_template<int> y;
  s_ptr_template<int> &x = y;

  constructible_from_sizeless cfs1(local_int8);
  constructible_from_sizeless cfs2 = local_int8;
#if __cplusplus >= 201103L
  constructible_from_sizeless cfs3{local_int8};
#endif

#if __cplusplus >= 201103L
  local_int8 = ([]() { return svint8_t(); })();
  local_int8 = ([]() -> svint8_t { return svint8_t(); })();
  auto fn1 = [&local_int8](svint8_t x) { local_int8 = x; };
  auto fn2 = [&local_int8](svint8_t *ptr) { *ptr = local_int8; };
#if __cplusplus >= 201703L
  auto fn3 = [a(return_int8())] {}; // expected-error {{field has sizeless type '__SVInt8_t'}}
#endif
  auto fn4 = [local_int8](svint8_t *ptr) { *ptr = local_int8; }; // expected-error {{by-copy capture of variable 'local_int8' with sizeless type 'svint8_t'}}

  for (auto x : local_int8) { // expected-error {{no viable 'begin' function available}}
  }
  for (auto x : wrapper<svint8_t>()) {
    (void)x;
  }
  for (const svint8_t &x : wrapper<svint8_t>()) { // expected-warning {{loop variable 'x' binds to a temporary value produced by a range of type 'wrapper<svint8_t>'}} expected-note {{use non-reference type}}
    (void)x;
  }
  for (const svint8_t x : wrapper<const svint8_t &>()) {
    (void)x;
  }
#endif
}

#if __cplusplus >= 201103L
svint8_t ret_bad_conv() { return explicit_conv(); } // expected-error {{no viable conversion from returned value of type 'explicit_conv' to function return type 'svint8_t'}}

#pragma clang diagnostic warning "-Wc++98-compat"

void incompat_init() { __attribute__((unused)) svint8_t foo = {}; } // expected-warning {{initializing 'svint8_t' (aka '__SVInt8_t') from an empty initializer list is incompatible with C++98}}

#endif
