// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv32 -DSPIRV32 %s

// __spirv_event_t is an opaque type: it cannot be initialized from, converted
// to, or cast from other types, nor used in arithmetic. It cannot be used in
// constant evaluation.
void foo() {
  int n = 100;
  __spirv_event_t v = 0; // expected-error {{cannot initialize a variable of type '__spirv_event_t' with an rvalue of type 'int'}}
  static_cast<__spirv_event_t>(n); // expected-error {{static_cast from 'int' to '__spirv_event_t' is not allowed}}
  reinterpret_cast<__spirv_event_t>(n); // expected-error {{reinterpret_cast from 'int' to '__spirv_event_t' is not allowed}}
  (void)(v + v); // expected-error {{invalid operands to binary expression ('__spirv_event_t' and '__spirv_event_t')}}
  int x(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__spirv_event_t'}}
  static_cast<int>(v); // expected-error {{static_cast from '__spirv_event_t' to 'int' is not allowed}}
  __spirv_event_t k;
  int *ip = (int *)k; // expected-error {{cannot cast from type '__spirv_event_t' to pointer type 'int *'}}
  constexpr __spirv_event_t e; // expected-error {{constexpr variable cannot have non-literal type 'const __spirv_event_t'}}
  (void)v; // Ok
}

template <__spirv_event_t V> void baz(); // expected-error {{a non-type template parameter cannot have type '__spirv_event_t'}}

// __spirv_event_t can be used as a function parameter, return type, template
// argument, and a struct field.
template <class T> void bar(T);
void use(__spirv_event_t r) { bar(r); }
__spirv_event_t make();
struct S { __spirv_event_t r; int a; };

// __spirv_event_t can be used as a template specialization argument, and the
// specialization is correctly selected over the primary template.
template <class T> struct TestSpecialization { static const int value = 0; };
template <> struct TestSpecialization<__spirv_event_t> { static const int value = 1; };
static_assert(TestSpecialization<__spirv_event_t>::value == 1, "specialization selected");
static_assert(TestSpecialization<int>::value == 0, "primary template selected");

// __spirv_event_t is copyable and moveable.
__spirv_event_t get();
void copy_and_move(__spirv_event_t a) {
  __spirv_event_t copyConstructed = a;
  __spirv_event_t moveConstructed = get();
  __spirv_event_t assigned;
  assigned = a;
  assigned = get();
}

// __spirv_event_t is an opaque, pointer-sized object type. It is not a scalar,
// class, union, enum, or fundamental type, and it is neither trivially copyable
// nor a literal type. These characteristics match OpenCL's event_t.
typedef __spirv_event_t __attribute__((address_space(7))) event_as7;
#ifdef SPIRV32
static_assert(sizeof(__spirv_event_t) == 4, "");
static_assert(alignof(__spirv_event_t) == 4, "");
static_assert(sizeof(event_as7) == 4, "");
static_assert(alignof(event_as7) == 4, "");
static_assert(sizeof(__spirv_event_t[4]) == 16, "");
#else
static_assert(sizeof(__spirv_event_t) == 8, "");
static_assert(alignof(__spirv_event_t) == 8, "");
static_assert(sizeof(event_as7) == 8, "");
static_assert(alignof(event_as7) == 8, "");
static_assert(sizeof(__spirv_event_t[4]) == 32, "");
#endif
static_assert(__is_object(__spirv_event_t), "");
static_assert(!__is_scalar(__spirv_event_t), "");
static_assert(!__is_arithmetic(__spirv_event_t), "");
static_assert(!__is_pointer(__spirv_event_t), "");
static_assert(!__is_compound(__spirv_event_t), "");
static_assert(!__is_fundamental(__spirv_event_t), "");
static_assert(!__is_class(__spirv_event_t), "");
static_assert(!__is_union(__spirv_event_t), "");
static_assert(!__is_enum(__spirv_event_t), "");
static_assert(!__is_pod(__spirv_event_t), "");
static_assert(!__is_aggregate(__spirv_event_t), "");
static_assert(!__is_trivial(__spirv_event_t), "");
static_assert(!__is_trivially_copyable(__spirv_event_t), "");
static_assert(__is_trivially_destructible(__spirv_event_t), "");
static_assert(!__is_standard_layout(__spirv_event_t), "");

