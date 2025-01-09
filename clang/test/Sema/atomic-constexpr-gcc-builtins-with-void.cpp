// RUN: %clang_cc1 -std=c++2c -verify %s

template <typename T = int, typename Y = int> constexpr auto test(int off) {
  T array[] = {T{1}, T{2}, T{3}, T{4}, T{5}};
  Y * ptr = static_cast<Y *>(&array[2]);
  Y * nptr = __atomic_add_fetch(&ptr, off, __ATOMIC_RELAXED); // #misalign
  return *static_cast<T*>(nptr);
}

constexpr auto tmp0 = test<int, int>(0);
static_assert(tmp0 == 3);

constexpr auto tmpmsz = test<int, int>(-sizeof(int));
static_assert(tmpmsz == 2);

constexpr auto tmp1 = test<int, int>(1);  // #tmp1
// expected-error-re@#tmp1 {{constexpr variable '{{tmp[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#tmp1 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 1 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto tmp2 = test<int, int>(2);  // #tmp2
// expected-error-re@#tmp2 {{constexpr variable '{{tmp[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#tmp2 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 2 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto tmp3 = test<int, int>(3);  // #tmp3
// expected-error-re@#tmp3 {{constexpr variable '{{tmp[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#tmp3 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 3 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto tmp4 = test<int, int>(sizeof(int));
static_assert(tmp4 == 4);

// =================
// GCC atomics work with void * pointers too, we must make sure they are aligned for size properly
constexpr auto void0 = test<int, void>(0);
static_assert(void0 == 3);

// negative
constexpr auto voidmsz = test<int, void>(-sizeof(int));
static_assert(voidmsz == 2);

constexpr auto void1 = test<int, void>(1);  // #void1
// expected-error-re@#void1 {{constexpr variable '{{void[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#void1 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 1 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto void2 = test<int, void>(2);  // #void2
// expected-error-re@#void2 {{constexpr variable '{{void[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#void2 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 2 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto void3 = test<int, void>(3);  // #void3
// expected-error-re@#void3 {{constexpr variable '{{void[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#void3 {{in call to '{{.+}}'}}
// expected-note-re@#misalign {{atomic pointer operation with argument 3 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}

constexpr auto void4 = test<int, void>(sizeof(int));
static_assert(void4 == 4);

// =================
// check overaligned types works too

struct alignas(16) special {
  int value;
};

static_assert(sizeof(special) == 16);

// negative
constexpr auto specm16 = test<special, void>(-16);
static_assert(specm16.value == 2);

// no change
constexpr auto spec0 = test<special, void>(0);
static_assert(spec0.value == 3);

constexpr auto spec1 = test<special, void>(1); // #spec1
// expected-error-re@#spec1 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec1 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 1 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec2 = test<special, void>(2); // #spec2
// expected-error-re@#spec2 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec2 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 2 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec3 = test<special, void>(3); // #spec3
// expected-error-re@#spec3 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec3 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 3 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec4 = test<special, void>(4); // #spec4
// expected-error-re@#spec4 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec4 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 4 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec5 = test<special, void>(5); // #spec5
// expected-error-re@#spec5 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec5 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 5 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec8 = test<special, void>(8); // #spec8
// expected-error-re@#spec8 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec8 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 8 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto spec15 = test<special, void>(15); // #spec15
// expected-error-re@#spec15 {{constexpr variable '{{spec[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#spec15 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument 15 not aligned to size of pointee type (sizeof 'special' is 16)}}

constexpr auto specm15 = test<special, void>(-15); // #specm15
// expected-error-re@#specm15 {{constexpr variable '{{specm[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#specm15 {{in call to '{{.+}}'}}
// expected-note@#misalign {{atomic pointer operation with argument -15 not aligned to size of pointee type (sizeof 'special' is 16)}}

// ===============
// any operations on incomplete types are disallowed
// (with exception of void* pointer pointing to complete type)

struct incomplete;

constexpr void * test2(void * ptr, int off) {
  incomplete * inc_ptr = static_cast<incomplete *>(ptr);
  incomplete * nptr = __atomic_add_fetch(&inc_ptr, off, __ATOMIC_RELAXED); // #incomplete
  return static_cast<void *>(nptr);
}

constexpr void * incmp0 = test2(nullptr, 0); // #incmp0
  
// expected-error-re@#incmp0 {{constexpr variable '{{incmp[0-9]+}}' must be initialized by a constant expression}}
// expected-note-re@#incmp0 {{in call to '{{.+}}'}}
// expected-error@#incomplete {{arithmetic on a pointer to an incomplete type 'incomplete'}}
