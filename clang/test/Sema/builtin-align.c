// RUN: %clang_cc1 -triple x86_64-linux-gnu -DALIGN_BUILTIN=__builtin_align_down -DRETURNS_BOOL=0 %s -fsyntax-only -verify -Wpedantic
// RUN: %clang_cc1 -triple x86_64-linux-gnu -DALIGN_BUILTIN=__builtin_align_up   -DRETURNS_BOOL=0 %s -fsyntax-only -verify -Wpedantic
// RUN: %clang_cc1 -triple x86_64-linux-gnu -DALIGN_BUILTIN=__builtin_is_aligned -DRETURNS_BOOL=1 %s -fsyntax-only -verify -Wpedantic

struct Aggregate {
  int i;
  int j;
};
enum Enum { EnumValue1,
            EnumValue2 };
typedef __SIZE_TYPE__ size_t;

void test_parameter_types(char *ptr, size_t size) {
  struct Aggregate agg;
  enum Enum e = EnumValue2;
  _Bool b = 0;

  // The first parameter can be any pointer or integer type:
  (void)ALIGN_BUILTIN(ptr, 4);
  (void)ALIGN_BUILTIN(size, 2);
  (void)ALIGN_BUILTIN(12345, 2);
  (void)ALIGN_BUILTIN(agg, 2);    // expected-error {{operand of type 'struct Aggregate' where arithmetic or pointer type is required}}
  (void)ALIGN_BUILTIN(e, 2);      // expected-error {{operand of type 'enum Enum' where arithmetic or pointer type is required}}
  (void)ALIGN_BUILTIN(b, 2);      // expected-error {{operand of type '_Bool' where arithmetic or pointer type is required}}
  (void)ALIGN_BUILTIN((int)e, 2); // but with a cast it is fine
  (void)ALIGN_BUILTIN((int)b, 2); // but with a cast it is fine

  // The second parameter must be an integer type (but not enum or _Bool):
  (void)ALIGN_BUILTIN(ptr, size);
  (void)ALIGN_BUILTIN(ptr, ptr);    // expected-error {{used type 'char *' where integer is required}}
  (void)ALIGN_BUILTIN(ptr, agg);    // expected-error {{used type 'struct Aggregate' where integer is required}}
  (void)ALIGN_BUILTIN(ptr, b);      // expected-error {{used type '_Bool' where integer is required}}
  (void)ALIGN_BUILTIN(ptr, e);      // expected-error {{used type 'enum Enum' where integer is required}}
  (void)ALIGN_BUILTIN(ptr, (int)e); // but with a cast enums are fine
  (void)ALIGN_BUILTIN(ptr, (int)b); // but with a cast booleans are fine

  (void)ALIGN_BUILTIN(ptr, size);
  (void)ALIGN_BUILTIN(size, size);
}

void test_result_unused(int i, int align) {
  // -Wunused-result does not trigger for macros so we can't use ALIGN_BUILTIN()
  // but need to explicitly call each function.
  __builtin_align_up(i, align);   // expected-warning{{ignoring return value of function declared with const attribute}}
  __builtin_align_down(i, align); // expected-warning{{ignoring return value of function declared with const attribute}}
  __builtin_is_aligned(i, align); // expected-warning{{ignoring return value of function declared with const attribute}}
  ALIGN_BUILTIN(i, align);        // no warning here
}

#define check_same_type(type1, type2) __builtin_types_compatible_p(type1, type2) && __builtin_types_compatible_p(type1 *, type2 *)

void test_return_type(void *ptr, int i, long l) {
  char array[32];
  __extension__ typedef typeof(ALIGN_BUILTIN(ptr, 4)) result_type_ptr;
  __extension__ typedef typeof(ALIGN_BUILTIN(i, 4)) result_type_int;
  __extension__ typedef typeof(ALIGN_BUILTIN(l, 4)) result_type_long;
  __extension__ typedef typeof(ALIGN_BUILTIN(array, 4)) result_type_char_array;
#if RETURNS_BOOL
  _Static_assert(check_same_type(_Bool, result_type_ptr), "Should return bool");
  _Static_assert(check_same_type(_Bool, result_type_int), "Should return bool");
  _Static_assert(check_same_type(_Bool, result_type_long), "Should return bool");
  _Static_assert(check_same_type(_Bool, result_type_char_array), "Should return bool");
#else
  _Static_assert(check_same_type(void *, result_type_ptr), "Should return void*");
  _Static_assert(check_same_type(int, result_type_int), "Should return int");
  _Static_assert(check_same_type(long, result_type_long), "Should return long");
  // Check that we can use the alignment builtins on on array types (result should decay)
  _Static_assert(check_same_type(char *, result_type_char_array),
                 "Using the builtins on an array should yield the decayed type");
#endif
}

void test_invalid_alignment_values(char *ptr, long *longptr, size_t align) {
  int x = 1;
  (void)ALIGN_BUILTIN(ptr, 2);
  (void)ALIGN_BUILTIN(longptr, 1024);
  (void)ALIGN_BUILTIN(x, 32);

  (void)ALIGN_BUILTIN(ptr, 0); // expected-error {{requested alignment must be 1 or greater}}
  (void)ALIGN_BUILTIN(ptr, 1);
#if RETURNS_BOOL
  // expected-warning@-2 {{checking whether a value is aligned to 1 byte is always true}}
#else
  // expected-warning@-4 {{aligning a value to 1 byte is a no-op}}
#endif
  (void)ALIGN_BUILTIN(ptr, 3); // expected-error {{requested alignment is not a power of 2}}
  (void)ALIGN_BUILTIN(x, 7);   // expected-error {{requested alignment is not a power of 2}}

  // check the maximum range for smaller types:
  __UINT8_TYPE__ c = ' ';

  (void)ALIGN_BUILTIN(c, 128);        // this is fine
  (void)ALIGN_BUILTIN(c, 256);        // expected-error {{requested alignment must be 128 or smaller}}
  (void)ALIGN_BUILTIN(x, 1ULL << 31); // this is also fine
  (void)ALIGN_BUILTIN(x, 1LL << 31);  // this is also fine
  __INT32_TYPE__ i32 = 3;
  __UINT32_TYPE__ u32 = 3;
  // Maximum is the same for int32 and uint32
  (void)ALIGN_BUILTIN(i32, 1ULL << 32);              // expected-error {{requested alignment must be 2147483648 or smaller}}
  (void)ALIGN_BUILTIN(u32, 1ULL << 32);              // expected-error {{requested alignment must be 2147483648 or smaller}}
  (void)ALIGN_BUILTIN(ptr, ((__int128)1) << 65);     // expected-error {{requested alignment must be 9223372036854775808 or smaller}}
  (void)ALIGN_BUILTIN(longptr, ((__int128)1) << 65); // expected-error {{requested alignment must be 9223372036854775808 or smaller}}

  const int bad_align = 8 + 1;
  (void)ALIGN_BUILTIN(ptr, bad_align); // expected-error {{requested alignment is not a power of 2}}
}

// Check that it can be used in constant expressions:
void constant_expression(int x) {
  _Static_assert(__builtin_is_aligned(1024, 512), "");
  _Static_assert(!__builtin_is_aligned(256, 512ULL), "");
  _Static_assert(__builtin_align_up(33, 32) == 64, "");
  _Static_assert(__builtin_align_down(33, 32) == 32, "");

  // But not if one of the arguments isn't constant:
  _Static_assert(ALIGN_BUILTIN(33, x) != 100, ""); // expected-error {{static assertion expression is not an integral constant expression}}
  _Static_assert(ALIGN_BUILTIN(x, 4) != 100, "");  // expected-error {{static assertion expression is not an integral constant expression}}
}

// Check that it is a constant expression that can be assigned to globals:
int global1 = __builtin_align_down(33, 8);
int global2 = __builtin_align_up(33, 8);
_Bool global3 = __builtin_is_aligned(33, 8);

extern void test_ptr(char *c);
char *test_array_and_fnptr(void) {
  char buf[1024];
  // The builtins should also work on arrays (decaying the return type)
  (void)(ALIGN_BUILTIN(buf, 16));
  // But not on functions and function pointers:
  (void)(ALIGN_BUILTIN(test_array_and_fnptr, 16));  // expected-error{{operand of type 'char *(void)' where arithmetic or pointer type is required}}
  (void)(ALIGN_BUILTIN(&test_array_and_fnptr, 16)); // expected-error{{operand of type 'char *(*)(void)' where arithmetic or pointer type is required}}
}
