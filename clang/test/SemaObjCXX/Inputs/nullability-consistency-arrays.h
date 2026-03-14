#include <stdarg.h>

void firstThingInTheFileThatNeedsNullabilityIsAnArray(int ints[]);
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif

int *secondThingInTheFileThatNeedsNullabilityIsAPointer;
#if !ARRAYS_CHECKED
// expected-warning@-2 {{pointer is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
// expected-note@-3 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-4 {{insert '_Nonnull' if the pointer should never be null}}
#endif

int *_Nonnull triggerConsistencyWarnings;

void test(
    int ints[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
    void *ptrs[], // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
#if ARRAYS_CHECKED
// expected-warning@-4 {{array parameter is missing a nullability type specifier}}
// expected-note@-5 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-6 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
    void **nestedPtrs[]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
// expected-note@-1 2 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 2 {{insert '_Nonnull' if the pointer should never be null}}
#if ARRAYS_CHECKED
// expected-warning@-4 {{array parameter is missing a nullability type specifier}}
// expected-note@-5 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-6 {{insert '_Nonnull' if the array parameter should never be null}}
#endif

void testArraysOK(
    int ints[_Nonnull],
    void *ptrs[_Nonnull], // expected-warning {{pointer is missing a nullability type specifier}}
// expected-note@-1 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 {{insert '_Nonnull' if the pointer should never be null}}
    void **nestedPtrs[_Nonnull]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
// expected-note@-1 2 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 2 {{insert '_Nonnull' if the pointer should never be null}}
void testAllOK(
    int ints[_Nonnull],
    void * _Nullable ptrs[_Nonnull],
    void * _Nullable * _Nullable nestedPtrs[_Nonnull]);

void testVAList(va_list ok); // no warning

#if __cplusplus
// Carefully construct a test case such that if a platform's va_list is an array
// or pointer type, it gets tested, but otherwise it does not.
template<class T, class F>
struct pointer_like_or { typedef F type; };
template<class T, class F>
struct pointer_like_or<T*, F> { typedef T *type; };
template<class T, class F>
struct pointer_like_or<T* const, F> { typedef T * const type; };
template<class T, class F>
struct pointer_like_or<T[], F> { typedef T type[]; };
template<class T, class F, unsigned size>
struct pointer_like_or<T[size], F> { typedef T type[size]; };

void testVAListWithNullability(
  pointer_like_or<va_list, void*>::type _Nonnull x); // no errors
#endif

void nestedArrays(int x[5][1]) {}
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
void nestedArraysOK(int x[_Nonnull 5][1]) {}

#if !__cplusplus
void staticOK(int x[static 5][1]){}
#endif

int globalArraysDoNotNeedNullability[5];

typedef int INTS[4];

void typedefTest(
    INTS x,
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
    INTS _Nonnull x2,
    _Nonnull INTS x3,
    INTS y[2],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
    INTS y2[_Nonnull 2]);


#pragma clang assume_nonnull begin
void testAssumeNonnull(
  int ints[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
  void *ptrs[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier}}
// expected-note@-3 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-4 {{insert '_Nonnull' if the array parameter should never be null}}
#endif
  void **nestedPtrs[]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
// expected-note@-1 2 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 2 {{insert '_Nonnull' if the pointer should never be null}}
#if ARRAYS_CHECKED
// expected-warning@-4 {{array parameter is missing a nullability type specifier}}
// expected-note@-5 {{insert '_Nullable' if the array parameter may be null}}
// expected-note@-6 {{insert '_Nonnull' if the array parameter should never be null}}
#endif

void testAssumeNonnullAllOK(
    int ints[_Nonnull],
    void * _Nullable ptrs[_Nonnull],
    void * _Nullable * _Nullable nestedPtrs[_Nonnull]);
void testAssumeNonnullAllOK2(
    int ints[_Nonnull],
    void * ptrs[_Nonnull], // backwards-compatibility
    void * _Nullable * _Nullable nestedPtrs[_Nonnull]);
#pragma clang assume_nonnull end
