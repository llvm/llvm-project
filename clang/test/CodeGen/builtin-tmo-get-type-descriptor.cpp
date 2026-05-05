// RUN: %clang_cc1 -triple x86_64-apple-darwin -fblocks -O0 -Rtmo-remarks -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fblocks -O0 -Rtmo-remarks -verify -fsyntax-only -fptrauth-intrinsics -fptrauth-calls %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -fblocks -O0 %s -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -emit-llvm -fblocks -O0 %s -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fptrauth-intrinsics -emit-llvm -fblocks -O0 %s -o - 2>&1 | FileCheck %s
// CHECK: %struct.Check = type { i8 }

_Static_assert(__has_builtin(__builtin_tmo_get_type_descriptor), "No type descriptor builtin");

typedef __INT8_TYPE__ uint8_t;
typedef unsigned __INT32_TYPE__ uint32_t;
typedef unsigned __INT64_TYPE__ uint64_t;

extern "C" void printf(const char *, uint32_t s) {}

#define SUMMARY_SHIFT                 32
#define SUMMARY_LAYOUT_SHIFT          (SUMMARY_SHIFT + 16)
#define SUMMARY_TYPE_FLAGS_SHIFT      (SUMMARY_SHIFT + 12)
#define SUMMARY_CALLSITE_FLAGS_SHIFT  (SUMMARY_SHIFT + 8)
#define SUMMARY_VERSION_SHIFT         (SUMMARY_SHIFT + 0)

#define SUMMARY_LAYOUT_BIT(b)         (1ULL << (SUMMARY_LAYOUT_SHIFT + b))
#define SUMMARY_TYPE_FLAGS_BIT(b)     (1ULL << (SUMMARY_TYPE_FLAGS_SHIFT + b))

#define SUMMARY_DATA_POINTER      SUMMARY_LAYOUT_BIT(0)
#define SUMMARY_STRUCT_POINTER    SUMMARY_LAYOUT_BIT(1)
#define SUMMARY_IMMUTABLE_POINTER SUMMARY_LAYOUT_BIT(2)
#define SUMMARY_ANONYMOUS_POINTER SUMMARY_LAYOUT_BIT(3)
#define SUMMARY_REFERENCE_COUNT   SUMMARY_LAYOUT_BIT(4)
#define SUMMARY_RESOURCE_HANDLE   SUMMARY_LAYOUT_BIT(5)
#define SUMMARY_SPATIAL_BOUNDS    SUMMARY_LAYOUT_BIT(6)
#define SUMMARY_TAINTED_DATA      SUMMARY_LAYOUT_BIT(7)
#define SUMMARY_GENERIC_DATA      SUMMARY_LAYOUT_BIT(8)

#define SUMMARY_IS_POLYMORPHIC    SUMMARY_TYPE_FLAGS_BIT(0)
#define SUMMARY_HAS_MIXED_UNIONS  SUMMARY_TYPE_FLAGS_BIT(1)
#define SUMMARY_FIXED_SIZE        (1 << 20)
#define SUMMARY_VERSION           (0x3 << 0)
#define SUMMARY_MASK              (0xffffffff00000000)
#define SUMMARY_MASK_LAYOUT       (0xffff000000000000)
#define SUMMARY_MASK_FLAGS        (0x0000f00000000000)
#define SUMMARY_MASK_CALLSITE     (0x00000f0000000000)


#define GET_DESCRIPTOR(e) __builtin_tmo_get_type_descriptor(e)
#define _GET_DESCRIPTOR_BITS(e, t, m) (static_cast<t>(e) & m)
#define GET_SUMMARY(e) \
  (_GET_DESCRIPTOR_BITS(e, uint32_t, SUMMARY_MASK) >> SUMMARY_SHIFT)
#define GET_HASH(e) _GET_DESCRIPTOR_BITS(e, uint32_t, ~SUMMARY_MASK)

template<typename T, uint64_t M, uint64_t C>
struct Check {
  static constexpr const uint64_t descr = GET_DESCRIPTOR(T); // #Check_descr
  static constexpr const uint64_t info = descr & M; // #Check_info
  static_assert((info ^ C) == 0, "check failed"); // #Check_static_assert
};

template<typename T, uint64_t C>
using CheckLayout = Check<T, SUMMARY_MASK_LAYOUT, C>;

template<typename T, uint64_t C>
using CheckFlags = Check<T, SUMMARY_MASK_FLAGS, C>;

template<typename T0, typename T1, bool Eq>
struct CompareHash {
  static constexpr const uint64_t d0 = GET_DESCRIPTOR(T0); // #CompareHashT0
  static constexpr const uint64_t d1 = GET_DESCRIPTOR(T1); // #CompareHashT1
  static_assert((d0 == d1) == Eq, "comparison failed"); // #ComparisonStaticAssert
};

struct S0 {
  int *d;
};
CheckLayout<S0, SUMMARY_DATA_POINTER> t0;
// expected-note@-1 {{in instantiation of template class 'Check<S0, 18446462598732840960, 281474976710656>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S0' as 281476107670517. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1130959861 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S0, 18446462598732840960, 281474976710656>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S0, 18446462598732840960, 281474976710656>::descr' requested here}}

CheckFlags<S0, 0> t0f;
// expected-note@-1 {{in instantiation of template class 'Check<S0, 263882790666240, 0>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S0' as 281476107670517. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1130959861 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S0, 263882790666240, 0>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S0, 263882790666240, 0>::descr' requested here}}

struct S1 {
  void *p;
};
CheckLayout<S1, SUMMARY_ANONYMOUS_POINTER> t1;
// expected-note@-1 {{in instantiation of template class 'Check<S1, 18446462598732840960, 2251799813685248>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S1' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S1, 18446462598732840960, 2251799813685248>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S1, 18446462598732840960, 2251799813685248>::descr' requested here}}

struct S2 {
  struct S1 *p;
};
CheckLayout<S2, SUMMARY_STRUCT_POINTER> t2;
// expected-note@-1 {{in instantiation of template class 'Check<S2, 18446462598732840960, 562949953421312>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S2' as 562952428289801. Type semantics: { "Summary": { "LayoutSemantics": [ "StructPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2474868489 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S2, 18446462598732840960, 562949953421312>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S2, 18446462598732840960, 562949953421312>::descr' requested here}}

struct S3a {
  const void *p;
};
struct S3b {
  void *const p;
};
CheckLayout<S3a, SUMMARY_ANONYMOUS_POINTER | SUMMARY_IMMUTABLE_POINTER> t3a;
// expected-note@-1 {{in instantiation of template class 'Check<S3a, 18446462598732840960, 3377699720527872>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S3a' as 3377702818697837. Type semantics: { "Summary": { "LayoutSemantics": [ "ImmutablePointer", "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3098169965 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S3a, 18446462598732840960, 3377699720527872>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S3a, 18446462598732840960, 3377699720527872>::descr' requested here}}

CheckLayout<S3b, SUMMARY_ANONYMOUS_POINTER | SUMMARY_IMMUTABLE_POINTER> t3b; // #t3b
// expected-note@#t3b {{in instantiation of template class 'Check<S3b, 18446462598732840960, 3377699720527872>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S3b' as 3377702818697837. Type semantics: { "Summary": { "LayoutSemantics": [ "ImmutablePointer", "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3098169965 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S3b, 18446462598732840960, 3377699720527872>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S3b, 18446462598732840960, 3377699720527872>::descr' requested here}}

struct S4 {
  char d[8];
};
CheckLayout<S4, SUMMARY_GENERIC_DATA> t4; // #t4
// expected-note@#t4 {{in instantiation of template class 'Check<S4, 18446462598732840960, 72057594037927936>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S4' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S4, 18446462598732840960, 72057594037927936>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S4, 18446462598732840960, 72057594037927936>::descr' requested here}}

struct S5 {
  union {
    void *pointer;
    uint64_t data;
  };
};
CheckLayout<S5, SUMMARY_GENERIC_DATA | SUMMARY_ANONYMOUS_POINTER> t5; // #t5
// expected-note@#t5 {{in instantiation of template class 'Check<S5, 18446462598732840960, 74309393851613184>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S5' as 74344580619983834. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2396281818 }}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S5, 18446462598732840960, 74309393851613184>::descr' requested here}}

CheckFlags<S5, SUMMARY_HAS_MIXED_UNIONS> t5f; // #t5f
// expected-note@#t5f {{in instantiation of template class 'Check<S5, 263882790666240, 35184372088832>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S5' as 74344580619983834. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2396281818 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S5, 263882790666240, 35184372088832>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S5, 263882790666240, 35184372088832>::descr' requested here}}

struct S6 {
  virtual void f() = 0;
  uint64_t d;
};
CheckLayout<S5, SUMMARY_GENERIC_DATA | SUMMARY_ANONYMOUS_POINTER> t6;
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S5, 18446462598732840960, 74309393851613184>::info' requested here}}

CheckFlags<S6, SUMMARY_IS_POLYMORPHIC> t6f; // #t6f
// expected-note@#t6f {{in instantiation of template class 'Check<S6, 263882790666240, 17592186044416>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S6' as 74326990272095183. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "IsPolymorphic" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4234437583 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S6, 263882790666240, 17592186044416>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S6, 263882790666240, 17592186044416>::descr' requested here}}

struct S7 {
  struct S0 s0;
  struct S1 s1;
  struct S5 s5;
  virtual void f() = 0;
};
CheckLayout<S7, SUMMARY_DATA_POINTER | SUMMARY_ANONYMOUS_POINTER | SUMMARY_GENERIC_DATA> t7; // #t7
// expected-note@#t7 {{in instantiation of template class 'Check<S7, 18446462598732840960, 74590868828323840>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S7' as 74643647262773326. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "AnonymousPointer", "GenericData" ], "TypeFlags": [ "IsPolymorphic", "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1876316238 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S7, 18446462598732840960, 74590868828323840>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S7, 18446462598732840960, 74590868828323840>::descr' requested here}}

CheckFlags<S7, SUMMARY_HAS_MIXED_UNIONS | SUMMARY_IS_POLYMORPHIC> t7f; // #t7f
// expected-note@#t7f {{in instantiation of template class 'Check<S7, 263882790666240, 52776558133248>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S7' as 74643647262773326. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "AnonymousPointer", "GenericData" ], "TypeFlags": [ "IsPolymorphic", "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1876316238 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S7, 263882790666240, 52776558133248>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S7, 263882790666240, 52776558133248>::descr' requested here}}

struct S8a {
  int *p;
  uint64_t u;
};
struct S8b {
  int *c;
  uint32_t a;
  uint32_t b;
};
struct S8c {
  char *c;
  char d0[2];
  char d1[4];
  char d2[2];
};
struct S8d {
  uint64_t u;
  int *p;
};
CompareHash<S8a, S8b, true> t8ca; // #t8ca
// expected-note@#t8ca {{in instantiation of template class 'CompareHash<S8a, S8b, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S8a' as 72339073273557324. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4258918732 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8b, true>::d0' requested here}}
// expected-note@#t8ca {{in instantiation of template class 'CompareHash<S8a, S8b, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S8b' as 72339073273557324. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4258918732 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8b, true>::d1' requested here}}

CompareHash<S8a, S8c, true> t8cb; // #t8cb
// expected-note@#t8cb {{in instantiation of template class 'CompareHash<S8a, S8c, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S8a' as 72339073273557324. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4258918732 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8c, true>::d0' requested here}}
// expected-note@#t8cb {{in instantiation of template class 'CompareHash<S8a, S8c, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S8c' as 72339073273557324. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4258918732 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8c, true>::d1' requested here}}
CompareHash<S8a, S8d, false> t8cc; // #t8cc
// expected-note@#t8cc {{in instantiation of template class 'CompareHash<S8a, S8d, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S8a' as 72339073273557324. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 4258918732 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8d, false>::d0' requested here}}
// expected-note@#t8cc {{in instantiation of template class 'CompareHash<S8a, S8d, false>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S8d' as 72339070195402188. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer", "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1180763596 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a, S8d, false>::d1' requested here}}
struct ForwardStruct;

CompareHash<void *, const void *, false> th0; // #th0
// expected-note@#th0 {{in instantiation of template class 'CompareHash<void *, const void *, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'void *' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, const void *, false>::d0' requested here}}
// expected-note@#th0 {{in instantiation of template class 'CompareHash<void *, const void *, false>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'const void *' as 3377702818697837. Type semantics: { "Summary": { "LayoutSemantics": [ "ImmutablePointer", "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3098169965 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, const void *, false>::d1' requested here}}
CompareHash<void *, void const*, false> th1;

CompareHash<void *, void **, true> th2; // #th2
// expected-note@#th2 {{in instantiation of template class 'CompareHash<void *, void **, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'void *' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, void **, true>::d0' requested here}}
// expected-note@#th2 {{in instantiation of template class 'CompareHash<void *, void **, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'void **' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, void **, true>::d1' requested here}}

CompareHash<void *, int *, false> th3; // #th3
// expected-note@#th3 {{in instantiation of template class 'CompareHash<void *, int *, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'void *' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, int *, false>::d0' requested here}}
// expected-note@#th3 {{in instantiation of template class 'CompareHash<void *, int *, false>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'int *' as 281476107670517. Type semantics: { "Summary": { "LayoutSemantics": [ "DataPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1130959861 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<void *, int *, false>::d1' requested here}}

CompareHash<uint64_t, uint32_t[2], true> th4; // #th4
// expected-note@#th4 {{in instantiation of template class 'CompareHash<unsigned long long, unsigned int[2], true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'unsigned long long' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<unsigned long long, unsigned int[2], true>::d0' requested here}}
// expected-note@#th4 {{in instantiation of template class 'CompareHash<unsigned long long, unsigned int[2], true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'unsigned int[2]' as 72057594041155351. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3227415 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<unsigned long long, unsigned int[2], true>::d1' requested here}}

CompareHash<ForwardStruct *, void *, false> th5; // #th5
// expected-note@#th5 {{in instantiation of template class 'CompareHash<ForwardStruct *, void *, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'ForwardStruct *' as 562952428289801. Type semantics: { "Summary": { "LayoutSemantics": [ "StructPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2474868489 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<ForwardStruct *, void *, false>::d0' requested here}}
// expected-note@#th5 {{in instantiation of template class 'CompareHash<ForwardStruct *, void *, false>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'void *' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<ForwardStruct *, void *, false>::d1' requested here}}

CompareHash<S8a *, ForwardStruct *, true> th6; // #th6
// expected-note@#th6 {{in instantiation of template class 'CompareHash<S8a *, ForwardStruct *, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S8a *' as 562952428289801. Type semantics: { "Summary": { "LayoutSemantics": [ "StructPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2474868489 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a *, ForwardStruct *, true>::d0' requested here}}
// expected-note@#th6 {{in instantiation of template class 'CompareHash<S8a *, ForwardStruct *, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'ForwardStruct *' as 562952428289801. Type semantics: { "Summary": { "LayoutSemantics": [ "StructPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2474868489 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S8a *, ForwardStruct *, true>::d1' requested here}}

struct S9a {
  union {
    char d[16];
    struct {
      char c[4];
      void *p;
    } __attribute__((packed));
  };
};
CheckFlags<S9a, SUMMARY_HAS_MIXED_UNIONS> t9af; // #t9af
// expected-note@#t9af {{in instantiation of template class 'Check<S9a, 263882790666240, 35184372088832>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S9a' as 74344578286568980. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer", "GenericData" ], "TypeFlags": [ "HasMixedUnions" ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 62866964 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S9a, 263882790666240, 35184372088832>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S9a, 263882790666240, 35184372088832>::descr' requested here}}

struct S10 {
  void (*f)(int);
};
CheckLayout<S10, SUMMARY_ANONYMOUS_POINTER> t10; // #t10
// expected-note@#t10 {{in instantiation of template class 'Check<S10, 18446462598732840960, 2251799813685248>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'S10' as 2251802906997560. Type semantics: { "Summary": { "LayoutSemantics": [ "AnonymousPointer" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 3093312312 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<S10, 18446462598732840960, 2251799813685248>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<S10, 18446462598732840960, 2251799813685248>::descr' requested here}}

struct S11a {
  int a[];
};
struct S11b {
};
struct S11c {
  union {
    int a[];
    char b[];
    void *p[];
  };
};
struct S11d {
  int a;
  int b[];
};
struct S11e {
  int a;
};
CompareHash<S11a, S11b, true> th7; // #th7
// expected-note@#th7 {{in instantiation of template class 'CompareHash<S11a, S11b, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S11a' as 170574065. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 170574065 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11b, true>::d0' requested here}}
// expected-note@#th7 {{in instantiation of template class 'CompareHash<S11a, S11b, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S11b' as 170574065. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 170574065 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11b, true>::d1' requested here}}

CompareHash<S11a, S11c, true> th8; // #th8
// expected-note@#th8 {{in instantiation of template class 'CompareHash<S11a, S11c, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S11a' as 170574065. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 170574065 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11c, true>::d0' requested here}}
// expected-note@#th8 {{in instantiation of template class 'CompareHash<S11a, S11c, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S11c' as 170574065. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 170574065 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11c, true>::d1' requested here}}

CompareHash<S11d, S11e, true> th9; // #th9
// expected-note@#th9 {{in instantiation of template class 'CompareHash<S11d, S11e, true>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S11d' as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11d, S11e, true>::d0' requested here}}
// expected-note@#th9 {{in instantiation of template class 'CompareHash<S11d, S11e, true>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S11e' as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11d, S11e, true>::d1' requested here}}

CompareHash<S11a, S11d, false> th10; // #th10
// expected-note@#th10 {{in instantiation of template class 'CompareHash<S11a, S11d, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'S11a' as 170574065. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 170574065 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11d, false>::d0' requested here}}
// expected-note@#th10 {{in instantiation of template class 'CompareHash<S11a, S11d, false>' requested here}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'S11d' as 72057595422605840. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 1384677904 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<S11a, S11d, false>::d1' requested here}}

// rdar://104683319
struct TypeA_104683319 {
  uint8_t d[1 << 22];
};

struct TypeB_104683319 {
  uint8_t d[1 << 22];
};

struct TypeC_104683319 {
  uint8_t d[1 << 21];
};

CompareHash<TypeA_104683319, TypeB_104683319, false> th1_104683319; // #th1_104683319
// expected-note@#th1_104683319 {{in instantiation of template class 'CompareHash<TypeA_104683319, TypeB_104683319, false>' requested here}}
// expected-remark@#CompareHashT0 {{__builtin_tmo_get_type_descriptor reported 'TypeA_104683319' as 2646530002. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2646530002 }}}
// expected-note@#th1_104683319 {{in instantiation of template class 'CompareHash<TypeA_104683319, TypeB_104683319, false>' requested here}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<TypeA_104683319, TypeB_104683319, false>::d0' requested her}}
// expected-remark@#CompareHashT1 {{__builtin_tmo_get_type_descriptor reported 'TypeB_104683319' as 865981442. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 865981442 }}}
// expected-note@#ComparisonStaticAssert {{in instantiation of static data member 'CompareHash<TypeA_104683319, TypeB_104683319, false>::d1' requested her}}

CheckLayout<TypeA_104683319, 0> t1_104683319; // #t1_104683319
// expected-note@#t1_104683319 {{in instantiation of template class 'Check<TypeA_104683319, 18446462598732840960, 0>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'TypeA_104683319' as 2646530002. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 2646530002 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<TypeA_104683319, 18446462598732840960, 0>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<TypeA_104683319, 18446462598732840960, 0>::descr' requested here}}

CheckLayout<TypeB_104683319, 0> t2_104683319; // #t2_104683319
// expected-note@#t2_104683319 {{in instantiation of template class 'Check<TypeB_104683319, 18446462598732840960, 0>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'TypeB_104683319' as 865981442. Type semantics: { "Summary": { "LayoutSemantics": [ ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 865981442 }}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<TypeB_104683319, 18446462598732840960, 0>::info' requested here}}
// expected-note@#Check_info {{in instantiation of static data member 'Check<TypeB_104683319, 18446462598732840960, 0>::descr' requested here}}

CheckLayout<TypeC_104683319, SUMMARY_GENERIC_DATA> t3_104683319; // #t3_104683319
// expected-note@#t3_104683319 {{in instantiation of template class 'Check<TypeC_104683319, 18446462598732840960, 72057594037927936>' requested here}}
// expected-remark@#Check_descr {{__builtin_tmo_get_type_descriptor reported 'TypeC_104683319' as 72057594228667364. Type semantics: { "Summary": { "LayoutSemantics": [ "GenericData" ], "TypeFlags": [ ], "TypeKind": "KindC", "CallsiteFlags": [ ] }, "TypeHash": 190739428 }}} 
// expected-note@#Check_info {{in instantiation of static data member 'Check<TypeC_104683319, 18446462598732840960, 72057594037927936>::descr' requested here}}
// expected-note@#Check_static_assert {{in instantiation of static data member 'Check<TypeC_104683319, 18446462598732840960, 72057594037927936>::info' requested here}}
