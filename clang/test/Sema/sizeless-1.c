// RUN: %clang_cc1 -fsyntax-only -verify -Wall -W -Wno-comment -Wno-strict-prototypes -triple arm64-linux-gnu -target-feature +sve -std=c90 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -W -Wno-strict-prototypes -triple arm64-linux-gnu -target-feature +sve -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -W -Wno-strict-prototypes -triple arm64-linux-gnu -target-feature +sve -std=gnu11 %s

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

int alignof_int8 = _Alignof(svint8_t);                // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}}
int alignof_int8_var = _Alignof(*extern_int8_ptr);    // expected-error {{invalid application of 'alignof' to sizeless type 'svint8_t'}} expected-warning {{GNU extension}}
int alignof_int8_var_ptr = _Alignof(extern_int8_ptr); // expected-warning {{GNU extension}}

void pass_int8(svint8_t); // expected-note {{passing argument to parameter here}}

svint8_t return_int8(void);

typedef svint8_t vec_int8_a __attribute__((vector_size(64)));    // expected-error {{invalid vector element type}}
typedef svint8_t vec_int8_b __attribute__((ext_vector_type(4))); // expected-error {{invalid vector element type}}

void dump(const volatile void *);

void __attribute__((overloadable)) overf(svint8_t);
void __attribute__((overloadable)) overf(svint16_t);

void __attribute__((overloadable)) overf8(svint8_t); // expected-note + {{not viable}}
void __attribute__((overloadable)) overf8(int);      // expected-note + {{not viable}}

void __attribute__((overloadable)) overf16(svint16_t); // expected-note + {{not viable}}
void __attribute__((overloadable)) overf16(int);       // expected-note + {{not viable}}

void noproto();
void varargs(int, ...);

void unused(void) {
  svint8_t unused_int8; // expected-warning {{unused}}
}

struct incomplete_struct *incomplete_ptr;

typedef svint8_t sizeless_array[1]; // expected-error {{array has sizeless element type}}

void func(int sel) {
  static svint8_t static_int8; // expected-error {{non-local variable with sizeless type 'svint8_t'}}

  svint8_t local_int8;
  int8_typedef typedef_int8;
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

  local_int8, 0; // expected-warning {{left operand of comma operator has no effect}}

  0, local_int8; // expected-warning {{left operand of comma operator has no effect}} expected-warning {{expression result unused}}

  svint8_t init_int8 = local_int8;
  svint8_t bad_init_int8 = for; // expected-error {{expected expression}}

  int empty_brace_init_int = {}; // expected-error {{scalar initializer cannot be empty}}
  svint8_t empty_brace_init_int8 = {}; // expected-error {{initializer for sizeless type 'svint8_t' (aka '__SVInt8_t') cannot be empty}}
  svint8_t brace_init_int8 = {local_int8};
  svint8_t bad_brace_init_int8_1 = {local_int8, 0};    // expected-warning {{excess elements in initializer for indivisible sizeless type 'svint8_t'}}
  svint8_t bad_brace_init_int8_2 = {0};                // expected-error {{incompatible type 'int'}}
  svint8_t bad_brace_init_int8_3 = {local_int16};      // expected-error {{incompatible type 'svint16_t'}}
  svint8_t bad_brace_init_int8_4 = {[0] = local_int8}; // expected-error {{designator in initializer for indivisible sizeless type 'svint8_t'}}
  svint8_t bad_brace_init_int8_5 = {{local_int8}};     // expected-warning {{too many braces around initializer}}
  svint8_t bad_brace_init_int8_6 = {{local_int8, 0}};  // expected-warning {{too many braces around initializer}}

  const svint8_t const_int8 = local_int8; // expected-note {{declared const here}}
  const svint8_t uninit_const_int8;

  volatile svint8_t volatile_int8;

  const volatile svint8_t const_volatile_int8 = local_int8; // expected-note {{declared const here}}
  const volatile svint8_t uninit_const_volatile_int8;

  _Atomic svint8_t atomic_int8;      // expected-error {{_Atomic cannot be applied to sizeless type 'svint8_t'}}
  __restrict svint8_t restrict_int8; // expected-error {{requires a pointer or reference}}

  svint8_t array_int8[1];          // expected-error {{array has sizeless element type}}
  svint8_t array_int8_init[] = {}; // expected-error {{array has sizeless element type}}

  _Bool test_int8 = init_int8; // expected-error {{initializing '_Bool' with an expression of incompatible type 'svint8_t'}}

  int int_int8 = init_int8; // expected-error {{initializing 'int' with an expression of incompatible type 'svint8_t'}}

  init_int8 = local_int8;
  init_int8 = local_int16; // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'svint16_t'}}
  init_int8 = sel;         // expected-error {{assigning to 'svint8_t' (aka '__SVInt8_t') from incompatible type 'int'}}

  sel = local_int8; // expected-error {{assigning to 'int' from incompatible type 'svint8_t'}}

  local_int8 = (svint8_t)local_int8;
  local_int8 = (const svint8_t)local_int8;
  local_int8 = (svint8_t)local_int16; // expected-error {{used type 'svint8_t' (aka '__SVInt8_t') where arithmetic or pointer type is required}}
  local_int8 = (svint8_t)0;           // expected-error {{used type 'svint8_t' (aka '__SVInt8_t') where arithmetic or pointer type is required}}
  sel = (int)local_int8;              // expected-error {{operand of type 'svint8_t' (aka '__SVInt8_t') where arithmetic or pointer type is required}}

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
  init_int8 = sel ? init_int8 : typedef_int8;
  init_int8 = sel ? init_int8 : const_int8;
  init_int8 = sel ? volatile_int8 : const_int8;
  init_int8 = sel ? volatile_int8 : const_volatile_int8;

  pass_int8(local_int8);
  pass_int8(local_int16); // expected-error {{passing 'svint16_t' (aka '__SVInt16_t') to parameter of incompatible type 'svint8_t'}}

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

  noproto(local_int8);
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
  local_int8 &&init_int8;  // expected-error {{invalid operands to binary expression}}
  local_int8 || init_int8; // expected-error {{invalid operands to binary expression}}

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
  local_int8 && 0; // expected-error {{invalid operands to binary expression}}
  local_int8 || 0; // expected-error {{invalid operands to binary expression}}

  if (local_int8) { // expected-error {{statement requires expression of scalar type}}
  }
  while (local_int8) { // expected-error {{statement requires expression of scalar type}}
  }
  do { // expected-error {{statement requires expression of scalar type}}
  } while (local_int8);
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

#if __STDC_VERSION__ >= 201112L
void test_generic(void) {
  svint8_t local_int8;
  svint16_t local_int16;

  int a1[_Generic(local_int8, svint8_t : 1, svint16_t : 2, default : 3) == 1 ? 1 : -1];
  int a2[_Generic(local_int16, svint8_t : 1, svint16_t : 2, default : 3) == 2 ? 1 : -1];
  int a3[_Generic(0, svint8_t : 1, svint16_t : 2, default : 3) == 3 ? 1 : -1];
  (void)_Generic(0, svint8_t : 1, svint8_t : 2, default : 3); // expected-error {{type 'svint8_t' (aka '__SVInt8_t') in generic association compatible with previously specified type 'svint8_t'}} expected-note {{compatible type}}
}

void test_compound_literal(void) {
  svint8_t local_int8;

  (void)(svint8_t){local_int8};
}
#endif
