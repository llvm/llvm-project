// C++-specific checks for the alignment builtins
// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -std=c++11 -o - %s -fsyntax-only -verify

// Check that we don't crash when using dependent types in __builtin_align:
template <typename a, a b>
void *c(void *d) { // expected-note{{candidate template ignored}}
  return __builtin_align_down(d, b);
}

struct x {};
x foo;
void test(void *value) {
  c<int, 16>(value);
  c<struct x, foo>(value); // expected-error{{no matching function for call to 'c'}}
}

template <typename T, long Alignment, long ArraySize = 16>
void test_templated_arguments() {
  T array[ArraySize];                                                           // expected-error{{variable has incomplete type 'fwddecl'}}
  static_assert(__is_same(decltype(__builtin_align_up(array, Alignment)), T *), // expected-error{{requested alignment is not a power of 2}}
                "return type should be the decayed array type");
  static_assert(__is_same(decltype(__builtin_align_down(array, Alignment)), T *),
                "return type should be the decayed array type");
  static_assert(__is_same(decltype(__builtin_is_aligned(array, Alignment)), bool),
                "return type should be bool");
  T *x1 = __builtin_align_up(array, Alignment);
  T *x2 = __builtin_align_down(array, Alignment);
  bool x3 = __builtin_align_up(array, Alignment);
}

void test() {
  test_templated_arguments<int, 32>(); // fine
  test_templated_arguments<struct fwddecl, 16>();
  // expected-note@-1{{in instantiation of function template specialization 'test_templated_arguments<fwddecl, 16L, 16L>'}}
  // expected-note@-2{{forward declaration of 'fwddecl'}}
  test_templated_arguments<int, 7>(); // invalid alignment value
  // expected-note@-1{{in instantiation of function template specialization 'test_templated_arguments<int, 7L, 16L>'}}
}

template <typename T, long ArraySize>
void test_incorrect_alignment_without_instatiation(T value) {
  int array[32];
  static_assert(__is_same(decltype(__builtin_align_up(array, 31)), int *), // expected-error{{requested alignment is not a power of 2}}
                "return type should be the decayed array type");
  static_assert(__is_same(decltype(__builtin_align_down(array, 7)), int *), // expected-error{{requested alignment is not a power of 2}}
                "return type should be the decayed array type");
  static_assert(__is_same(decltype(__builtin_is_aligned(array, -1)), bool), // expected-error{{requested alignment must be 1 or greater}}
                "return type should be bool");
  __builtin_align_up(array);       // expected-error{{too few arguments to function call, expected 2, have 1}}
  __builtin_align_up(array, 31);   // expected-error{{requested alignment is not a power of 2}}
  __builtin_align_down(array, 31); // expected-error{{requested alignment is not a power of 2}}
  __builtin_align_up(array, 31);   // expected-error{{requested alignment is not a power of 2}}
  __builtin_align_up(value, 31);   // This shouldn't want since the type is dependent
  __builtin_align_up(value);       // Same here

  __builtin_align_up(array, sizeof(sizeof(value)) - 1); // expected-error{{requested alignment is not a power of 2}}
  __builtin_align_up(array, value); // no diagnostic as the alignment is value dependent.
  (void)__builtin_align_up(array, ArraySize); // The same above here
}

// The original fix for the issue above broke some legitimate code.
// Here is a regression test:
typedef __SIZE_TYPE__ size_t;
void *allocate_impl(size_t size);
template <typename T>
T *allocate() {
  constexpr size_t allocation_size =
      __builtin_align_up(sizeof(T), sizeof(void *));
  return static_cast<T *>(
      __builtin_assume_aligned(allocate_impl(allocation_size), sizeof(void *)));
}
struct Foo {
  int value;
};
void *test2() {
  return allocate<struct Foo>();
}

// Check that pointers-to-members cannot be used:
class MemPtr {
public:
  int data;
  void func();
  virtual void vfunc();
};
void test_member_ptr() {
  __builtin_align_up(&MemPtr::data, 64);    // expected-error{{operand of type 'int MemPtr::*' where arithmetic or pointer type is required}}
  __builtin_align_down(&MemPtr::func, 64);  // expected-error{{operand of type 'void (MemPtr::*)()' where arithmetic or pointer type is required}}
  __builtin_is_aligned(&MemPtr::vfunc, 64); // expected-error{{operand of type 'void (MemPtr::*)()' where arithmetic or pointer type is required}}
}

void test_references(Foo &i) {
  // Check that the builtins look at the referenced type rather than the reference itself.
  (void)__builtin_align_up(i, 64);                            // expected-error{{operand of type 'Foo' where arithmetic or pointer type is required}}
  (void)__builtin_align_up(static_cast<Foo &>(i), 64);        // expected-error{{operand of type 'Foo' where arithmetic or pointer type is required}}
  (void)__builtin_align_up(static_cast<const Foo &>(i), 64);  // expected-error{{operand of type 'const Foo' where arithmetic or pointer type is required}}
  (void)__builtin_align_up(static_cast<Foo &&>(i), 64);       // expected-error{{operand of type 'Foo' where arithmetic or pointer type is required}}
  (void)__builtin_align_up(static_cast<const Foo &&>(i), 64); // expected-error{{operand of type 'const Foo' where arithmetic or pointer type is required}}
  (void)__builtin_align_up(&i, 64);
}

// Check that constexpr wrapper functions can be constant-evaluated.
template <typename T>
constexpr bool wrap_is_aligned(T ptr, long align) {
  return __builtin_is_aligned(ptr, align);
  // expected-note@-1{{requested alignment -3 is not a positive power of two}}
  // expected-note@-2{{requested alignment 19 is not a positive power of two}}
  // expected-note@-3{{requested alignment must be 128 or less for type 'char'; 4194304 is invalid}}
}
template <typename T>
constexpr T wrap_align_up(T ptr, long align) {
  return __builtin_align_up(ptr, align);
  // expected-note@-1{{requested alignment -2 is not a positive power of two}}
  // expected-note@-2{{requested alignment 18 is not a positive power of two}}
  // expected-note@-3{{requested alignment must be 2147483648 or less for type 'int'; 8589934592 is invalid}}
  // expected-error@-4{{operand of type 'bool' where arithmetic or pointer type is required}}
}

template <typename T>
constexpr T wrap_align_down(T ptr, long align) {
  return __builtin_align_down(ptr, align);
  // expected-note@-1{{requested alignment -1 is not a positive power of two}}
  // expected-note@-2{{requested alignment 17 is not a positive power of two}}
  // expected-note@-3{{requested alignment must be 32768 or less for type 'short'; 1048576 is invalid}}
}

constexpr int a1 = wrap_align_up(22, 32);
static_assert(a1 == 32, "");
constexpr int a2 = wrap_align_down(22, 16);
static_assert(a2 == 16, "");
constexpr bool a3 = wrap_is_aligned(22, 32);
static_assert(!a3, "");
static_assert(wrap_align_down(wrap_align_up(22, 16), 32) == 32, "");
static_assert(wrap_is_aligned(wrap_align_down(wrap_align_up(22, 16), 32), 32), "");
static_assert(!wrap_is_aligned(wrap_align_down(wrap_align_up(22, 16), 32), 64), "");

constexpr long const_value(long l) { return l; }
// Check some invalid values during constant-evaluation
static_assert(wrap_align_down(1, const_value(-1)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_down(1, -1)'}}
static_assert(wrap_align_up(1, const_value(-2)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_up(1, -2)'}}
static_assert(wrap_is_aligned(1, const_value(-3)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_is_aligned(1, -3)'}}
static_assert(wrap_align_down(1, const_value(17)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_down(1, 17)'}}
static_assert(wrap_align_up(1, const_value(18)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_up(1, 18)'}}
static_assert(wrap_is_aligned(1, const_value(19)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_is_aligned(1, 19)'}}

// Check invalid values for smaller types:
static_assert(wrap_align_down(static_cast<short>(1), const_value(1 << 20)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_down(1, 1048576)'}}
// Check invalid boolean type
static_assert(wrap_align_up(static_cast<int>(1), const_value(1ull << 33)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_align_up(1, 8589934592)'}}
static_assert(wrap_is_aligned(static_cast<char>(1), const_value(1 << 22)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in call to 'wrap_is_aligned(1, 4194304)'}}

// Check invalid boolean type
static_assert(wrap_align_up(static_cast<bool>(1), const_value(1 << 21)), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{in instantiation of function template specialization 'wrap_align_up<bool>' requested here}}

// Check constant evaluation for pointers:
_Alignas(32) char align32array[128];
static_assert(&align32array[0] == &align32array[0], "");
// __builtin_align_up/down can be constant evaluated as a no-op for values
// that are known to have greater alignment:
static_assert(__builtin_align_up(&align32array[0], 32) == &align32array[0], "");
static_assert(__builtin_align_up(&align32array[0], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[0], 4) == __builtin_align_up(&align32array[0], 8), "");
// But it can not be evaluated if the alignment is greater than the minimum
// known alignment, since in that case the value might be the same if it happens
// to actually be aligned to 64 bytes at run time.
static_assert(&align32array[0] == __builtin_align_up(&align32array[0], 64), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{cannot constant evaluate the result of adjusting alignment to 64}}
static_assert(__builtin_align_up(&align32array[0], 64) == __builtin_align_up(&align32array[0], 64), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{cannot constant evaluate the result of adjusting alignment to 64}}

// However, we can compute in case the requested alignment is less than the
// base alignment:
static_assert(__builtin_align_up(&align32array[0], 4) == &align32array[0], "");
static_assert(__builtin_align_up(&align32array[1], 4) == &align32array[4], "");
static_assert(__builtin_align_up(&align32array[2], 4) == &align32array[4], "");
static_assert(__builtin_align_up(&align32array[3], 4) == &align32array[4], "");
static_assert(__builtin_align_up(&align32array[4], 4) == &align32array[4], "");
static_assert(__builtin_align_up(&align32array[5], 4) == &align32array[8], "");
static_assert(__builtin_align_up(&align32array[6], 4) == &align32array[8], "");
static_assert(__builtin_align_up(&align32array[7], 4) == &align32array[8], "");
static_assert(__builtin_align_up(&align32array[8], 4) == &align32array[8], "");

static_assert(__builtin_align_down(&align32array[0], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[1], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[2], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[3], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[4], 4) == &align32array[4], "");
static_assert(__builtin_align_down(&align32array[5], 4) == &align32array[4], "");
static_assert(__builtin_align_down(&align32array[6], 4) == &align32array[4], "");
static_assert(__builtin_align_down(&align32array[7], 4) == &align32array[4], "");
static_assert(__builtin_align_down(&align32array[8], 4) == &align32array[8], "");

// Achieving the same thing using casts to uintptr_t is not allowed:
static_assert((char *)((__UINTPTR_TYPE__)&align32array[7] & ~3) == &align32array[4], ""); // expected-error{{not an integral constant expression}}

static_assert(__builtin_align_down(&align32array[1], 4) == &align32array[0], "");
static_assert(__builtin_align_down(&align32array[1], 64) == &align32array[0], ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{cannot constant evaluate the result of adjusting alignment to 64}}

// Add some checks for __builtin_is_aligned:
static_assert(__builtin_is_aligned(&align32array[0], 32), "");
static_assert(__builtin_is_aligned(&align32array[4], 4), "");
// We cannot constant evaluate whether the array is aligned to > 32 since this
// may well be true at run time.
static_assert(!__builtin_is_aligned(&align32array[0], 64), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{cannot constant evaluate whether run-time alignment is at least 64}}

// However, if the alignment being checked is less than the minimum alignment of
// the base object we can check the low bits of the alignment:
static_assert(__builtin_is_aligned(&align32array[0], 4), "");
static_assert(!__builtin_is_aligned(&align32array[1], 4), "");
static_assert(!__builtin_is_aligned(&align32array[2], 4), "");
static_assert(!__builtin_is_aligned(&align32array[3], 4), "");
static_assert(__builtin_is_aligned(&align32array[4], 4), "");

// TODO: this should evaluate to true even though we can't evaluate the result
//  of __builtin_align_up() to a concrete value
static_assert(__builtin_is_aligned(__builtin_align_up(&align32array[0], 64), 64), ""); // expected-error{{not an integral constant expression}}
// expected-note@-1{{cannot constant evaluate the result of adjusting alignment to 64}}

// Check different source and alignment type widths are handled correctly.
static_assert(!__builtin_is_aligned(static_cast<signed long>(7), static_cast<signed short>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<signed short>(7), static_cast<signed long>(4)), "");
// Also check signed -- unsigned mismatch.
static_assert(!__builtin_is_aligned(static_cast<signed long>(7), static_cast<signed long>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<unsigned long>(7), static_cast<unsigned long>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<signed long>(7), static_cast<unsigned long>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<unsigned long>(7), static_cast<signed long>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<signed long>(7), static_cast<unsigned short>(4)), "");
static_assert(!__builtin_is_aligned(static_cast<unsigned short>(7), static_cast<signed long>(4)), "");

// Check the diagnostic message
_Alignas(void) char align_void_array[1]; // expected-error {{invalid application of '_Alignas' to an incomplete type 'void'}}
