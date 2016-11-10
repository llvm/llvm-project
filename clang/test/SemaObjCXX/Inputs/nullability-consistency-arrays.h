void firstThingInTheFileThatNeedsNullabilityIsAnArray(int ints[]);
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif

int *secondThingInTheFileThatNeedsNullabilityIsAPointer;
#if !ARRAYS_CHECKED
// expected-warning@-2 {{pointer is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif

int *_Nonnull triggerConsistencyWarnings;

void test(
    int ints[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
    void *ptrs[], // expected-warning {{pointer is missing a nullability type specifier}}
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
    void **nestedPtrs[]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif

void testArraysOK(
    int ints[_Nonnull],
    void *ptrs[_Nonnull], // expected-warning {{pointer is missing a nullability type specifier}}
    void **nestedPtrs[_Nonnull]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
void testAllOK(
    int ints[_Nonnull],
    void * _Nullable ptrs[_Nonnull],
    void * _Nullable * _Nullable nestedPtrs[_Nonnull]);

void nestedArrays(int x[5][1]) {}
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
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
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
    INTS _Nonnull x2,
    _Nonnull INTS x3,
    INTS y[2],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
    INTS y2[_Nonnull 2]);


#pragma clang assume_nonnull begin
void testAssumeNonnull(
  int ints[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
  void *ptrs[],
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
#endif
  void **nestedPtrs[]); // expected-warning 2 {{pointer is missing a nullability type specifier}}
#if ARRAYS_CHECKED
// expected-warning@-2 {{array parameter is missing a nullability type specifier (_Nonnull, _Nullable, or _Null_unspecified)}}
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
