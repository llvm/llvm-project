// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s

template<typename T> void used(T &) noexcept;

#define TEST_UNINIT(NAME, TYPE)                 \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_uninit() {                 \
    type_##NAME uninit;                         \
    used(uninit);                               \
  }

// Value initialization on scalars, aggregate initialization on aggregates.
#define TEST_BRACES(NAME, TYPE)                 \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_braces() {                 \
    type_##NAME braces = {};                    \
    used(braces);                               \
  }

#define TEST_CUSTOM(NAME, TYPE, ...)            \
  using type_##NAME = TYPE;                     \
  void test_##NAME##_custom() {                 \
    type_##NAME custom __VA_ARGS__;             \
    used(custom);                               \
  }

struct empty {};
struct small { char c; };
struct smallinit { char c = 42; };
struct smallpartinit { char c = 42, d; };
struct nullinit { char* null = nullptr; };
struct padded { char c; int i; };
struct paddednullinit { char c = 0; int i = 0; };
struct bitfield { int i : 4; int j : 2; };
struct bitfieldaligned { int i : 4; int : 0; int j : 2; };
struct big { unsigned a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z; };
struct arraytail { int i; int arr[]; };
struct tailpad { short s; char c; };
struct notlockfree { long long a[4]; };
struct semivolatile { int i; volatile int vi; };
struct semivolatileinit { int i = 0x11111111; volatile int vi = 0x11111111; };
struct base { virtual ~base(); };
struct derived : public base {};
struct virtualderived : public virtual base, public virtual derived {};
union matching { int i; float f; };
union matchingreverse { float f; int i; };
union unmatched { char c; int i; };
union unmatchedreverse { int i; char c; };
union unmatchedfp { float f; double d; };
enum emptyenum {};
enum smallenum { VALUE };

extern "C" {

TEST_UNINIT(char, char);
// CHECK-LABEL: @test_char_uninit()
// CHECK:       %uninit = alloca i8, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(char, char);
// CHECK-LABEL: @test_char_braces()
// CHECK:       %braces = alloca i8, align 1
// CHECK-NEXT:  store i8 0, i8* %braces, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(uchar, unsigned char);
// CHECK-LABEL: @test_uchar_uninit()
// CHECK:       %uninit = alloca i8, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(uchar, unsigned char);
// CHECK-LABEL: @test_uchar_braces()
// CHECK:       %braces = alloca i8, align 1
// CHECK-NEXT:  store i8 0, i8* %braces, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(schar, signed char);
// CHECK-LABEL: @test_schar_uninit()
// CHECK:       %uninit = alloca i8, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(schar, signed char);
// CHECK-LABEL: @test_schar_braces()
// CHECK:       %braces = alloca i8, align 1
// CHECK-NEXT:  store i8 0, i8* %braces, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(wchar_t, wchar_t);
// CHECK-LABEL: @test_wchar_t_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(wchar_t, wchar_t);
// CHECK-LABEL: @test_wchar_t_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(short, short);
// CHECK-LABEL: @test_short_uninit()
// CHECK:       %uninit = alloca i16, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(short, short);
// CHECK-LABEL: @test_short_braces()
// CHECK:       %braces = alloca i16, align 2
// CHECK-NEXT:  store i16 0, i16* %braces, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ushort, unsigned short);
// CHECK-LABEL: @test_ushort_uninit()
// CHECK:       %uninit = alloca i16, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(ushort, unsigned short);
// CHECK-LABEL: @test_ushort_braces()
// CHECK:       %braces = alloca i16, align 2
// CHECK-NEXT:  store i16 0, i16* %braces, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int, int);
// CHECK-LABEL: @test_int_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int, int);
// CHECK-LABEL: @test_int_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(unsigned, unsigned);
// CHECK-LABEL: @test_unsigned_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(unsigned, unsigned);
// CHECK-LABEL: @test_unsigned_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(long, long);
// CHECK-LABEL: @test_long_uninit()
// CHECK:       %uninit = alloca i64, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(long, long);
// CHECK-LABEL: @test_long_braces()
// CHECK:       %braces = alloca i64, align 8
// CHECK-NEXT:  store i64 0, i64* %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ulong, unsigned long);
// CHECK-LABEL: @test_ulong_uninit()
// CHECK:       %uninit = alloca i64, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(ulong, unsigned long);
// CHECK-LABEL: @test_ulong_braces()
// CHECK:       %braces = alloca i64, align 8
// CHECK-NEXT:  store i64 0, i64* %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(longlong, long long);
// CHECK-LABEL: @test_longlong_uninit()
// CHECK:       %uninit = alloca i64, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(longlong, long long);
// CHECK-LABEL: @test_longlong_braces()
// CHECK:       %braces = alloca i64, align 8
// CHECK-NEXT:  store i64 0, i64* %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(ulonglong, unsigned long long);
// CHECK-LABEL: @test_ulonglong_uninit()
// CHECK:       %uninit = alloca i64, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(ulonglong, unsigned long long);
// CHECK-LABEL: @test_ulonglong_braces()
// CHECK:       %braces = alloca i64, align 8
// CHECK-NEXT:  store i64 0, i64* %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int128, __int128);
// CHECK-LABEL: @test_int128_uninit()
// CHECK:       %uninit = alloca i128, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int128, __int128);
// CHECK-LABEL: @test_int128_braces()
// CHECK:       %braces = alloca i128, align 16
// CHECK-NEXT:  store i128 0, i128* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(uint128, unsigned __int128);
// CHECK-LABEL: @test_uint128_uninit()
// CHECK:       %uninit = alloca i128, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(uint128, unsigned __int128);
// CHECK-LABEL: @test_uint128_braces()
// CHECK:       %braces = alloca i128, align 16
// CHECK-NEXT:  store i128 0, i128* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)


TEST_UNINIT(fp16, __fp16);
// CHECK-LABEL: @test_fp16_uninit()
// CHECK:       %uninit = alloca half, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(fp16, __fp16);
// CHECK-LABEL: @test_fp16_braces()
// CHECK:       %braces = alloca half, align 2
// CHECK-NEXT:  store half 0xH0000, half* %braces, align 2
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(float, float);
// CHECK-LABEL: @test_float_uninit()
// CHECK:       %uninit = alloca float, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(float, float);
// CHECK-LABEL: @test_float_braces()
// CHECK:       %braces = alloca float, align 4
// CHECK-NEXT:  store float 0.000000e+00, float* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(double, double);
// CHECK-LABEL: @test_double_uninit()
// CHECK:       %uninit = alloca double, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(double, double);
// CHECK-LABEL: @test_double_braces()
// CHECK:       %braces = alloca double, align 8
// CHECK-NEXT:  store double 0.000000e+00, double* %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(longdouble, long double);
// CHECK-LABEL: @test_longdouble_uninit()
// CHECK:       %uninit = alloca x86_fp80, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(longdouble, long double);
// CHECK-LABEL: @test_longdouble_braces()
// CHECK:       %braces = alloca x86_fp80, align 16
// CHECK-NEXT:  store x86_fp80 0xK00000000000000000000, x86_fp80* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)


TEST_UNINIT(intptr, int*);
// CHECK-LABEL: @test_intptr_uninit()
// CHECK:       %uninit = alloca i32*, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(intptr, int*);
// CHECK-LABEL: @test_intptr_braces()
// CHECK:       %braces = alloca i32*, align 8
// CHECK-NEXT:  store i32* null, i32** %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(intptrptr, int**);
// CHECK-LABEL: @test_intptrptr_uninit()
// CHECK:       %uninit = alloca i32**, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(intptrptr, int**);
// CHECK-LABEL: @test_intptrptr_braces()
// CHECK:       %braces = alloca i32**, align 8
// CHECK-NEXT:  store i32** null, i32*** %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(function, void(*)());
// CHECK-LABEL: @test_function_uninit()
// CHECK:       %uninit = alloca void ()*, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(function, void(*)());
// CHECK-LABEL: @test_function_braces()
// CHECK:       %braces = alloca void ()*, align 8
// CHECK-NEXT:  store void ()* null, void ()** %braces, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(bool, bool);
// CHECK-LABEL: @test_bool_uninit()
// CHECK:       %uninit = alloca i8, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(bool, bool);
// CHECK-LABEL: @test_bool_braces()
// CHECK:       %braces = alloca i8, align 1
// CHECK-NEXT:  store i8 0, i8* %braces, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)


TEST_UNINIT(empty, empty);
// CHECK-LABEL: @test_empty_uninit()
// CHECK:       %uninit = alloca %struct.empty, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(empty, empty);
// CHECK-LABEL: @test_empty_braces()
// CHECK:       %braces = alloca %struct.empty, align 1
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(small, small);
// CHECK-LABEL: @test_small_uninit()
// CHECK:       %uninit = alloca %struct.small, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(small, small);
// CHECK-LABEL: @test_small_braces()
// CHECK:       %braces = alloca %struct.small, align 1
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 1 %{{.*}}, i8 0, i64 1, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

  TEST_CUSTOM(small, small, { 42 });
// CHECK-LABEL: @test_small_custom()
// CHECK:       %custom = alloca %struct.small, align 1
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallinit, smallinit);
// CHECK-LABEL: @test_smallinit_uninit()
// CHECK:       %uninit = alloca %struct.smallinit, align 1
// CHECK-NEXT:  call void @{{.*}}smallinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(smallinit, smallinit);
// CHECK-LABEL: @test_smallinit_braces()
// CHECK:       %braces = alloca %struct.smallinit, align 1
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallinit, %struct.smallinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallinit, smallinit, { 100 });
// CHECK-LABEL: @test_smallinit_custom()
// CHECK:       %custom = alloca %struct.smallinit, align 1
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallinit, %struct.smallinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 100, i8* %[[C]], align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallpartinit, smallpartinit);
// CHECK-LABEL: @test_smallpartinit_uninit()
// CHECK:       %uninit = alloca %struct.smallpartinit, align 1
// CHECK-NEXT:  call void @{{.*}}smallpartinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(smallpartinit, smallpartinit);
// CHECK-LABEL: @test_smallpartinit_braces()
// CHECK:       %braces = alloca %struct.smallpartinit, align 1
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align 1
// CHECK-NEXT:  %[[D:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store i8 0, i8* %[[D]], align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallpartinit, smallpartinit, { 100, 42 });
// CHECK-LABEL: @test_smallpartinit_custom()
// CHECK:       %custom = alloca %struct.smallpartinit, align 1
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 100, i8* %[[C]], align 1
// CHECK-NEXT:  %[[D:[^ ]*]] = getelementptr inbounds %struct.smallpartinit, %struct.smallpartinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store i8 42, i8* %[[D]], align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(nullinit, nullinit);
// CHECK-LABEL: @test_nullinit_uninit()
// CHECK:       %uninit = alloca %struct.nullinit, align 8
// CHECK-NEXT:  call void @{{.*}}nullinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(nullinit, nullinit);
// CHECK-LABEL: @test_nullinit_braces()
// CHECK:       %braces = alloca %struct.nullinit, align 8
// CHECK-NEXT:  %[[N:[^ ]*]] = getelementptr inbounds %struct.nullinit, %struct.nullinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8* null, i8** %[[N]], align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(nullinit, nullinit, { (char*)"derp" });
// CHECK-LABEL: @test_nullinit_custom()
// CHECK:       %custom = alloca %struct.nullinit, align 8
// CHECK-NEXT:  %[[N:[^ ]*]] = getelementptr inbounds %struct.nullinit, %struct.nullinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8* getelementptr inbounds {{.*}}, i8** %[[N]], align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(padded, padded);
// CHECK-LABEL: @test_padded_uninit()
// CHECK:       %uninit = alloca %struct.padded, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(padded, padded);
// CHECK-LABEL: @test_padded_braces()
// CHECK:       %braces = alloca %struct.padded, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(padded, padded, { 42, 13371337 });
// CHECK-LABEL: @test_padded_custom()
// CHECK:       %custom = alloca %struct.padded, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(paddednullinit, paddednullinit);
// CHECK-LABEL: @test_paddednullinit_uninit()
// CHECK:       %uninit = alloca %struct.paddednullinit, align 4
// CHECK-NEXT:  call void @{{.*}}paddednullinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(paddednullinit, paddednullinit);
// CHECK-LABEL: @test_paddednullinit_braces()
// CHECK:       %braces = alloca %struct.paddednullinit, align 4
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i8 0, i8* %[[C]], align 4
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store i32 0, i32* %[[I]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(paddednullinit, paddednullinit, { 42, 13371337 });
// CHECK-LABEL: @test_paddednullinit_custom()
// CHECK:       %custom = alloca %struct.paddednullinit, align 4
// CHECK-NEXT:  %[[C:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i8 42, i8* %[[C]], align 4
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.paddednullinit, %struct.paddednullinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store i32 13371337, i32* %[[I]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bitfield, bitfield);
// CHECK-LABEL: @test_bitfield_uninit()
// CHECK:       %uninit = alloca %struct.bitfield, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(bitfield, bitfield);
// CHECK-LABEL: @test_bitfield_braces()
// CHECK:       %braces = alloca %struct.bitfield, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bitfield, bitfield, { 4, 1 });
// CHECK-LABEL: @test_bitfield_custom()
// CHECK:       %custom = alloca %struct.bitfield, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bitfieldaligned, bitfieldaligned);
// CHECK-LABEL: @test_bitfieldaligned_uninit()
// CHECK:       %uninit = alloca %struct.bitfieldaligned, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(bitfieldaligned, bitfieldaligned);
// CHECK-LABEL: @test_bitfieldaligned_braces()
// CHECK:       %braces = alloca %struct.bitfieldaligned, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bitfieldaligned, bitfieldaligned, { 4, 1  });
// CHECK-LABEL: @test_bitfieldaligned_custom()
// CHECK:       %custom = alloca %struct.bitfieldaligned, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(big, big);
// CHECK-LABEL: @test_big_uninit()
// CHECK:       %uninit = alloca %struct.big, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(big, big);
// CHECK-LABEL: @test_big_braces()
// CHECK:       %braces = alloca %struct.big, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 104, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(big, big, { 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA });
// CHECK-LABEL: @test_big_custom()
// CHECK:       %custom = alloca %struct.big, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 -86, i64 104, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(arraytail, arraytail);
// CHECK-LABEL: @test_arraytail_uninit()
// CHECK:       %uninit = alloca %struct.arraytail, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(arraytail, arraytail);
// CHECK-LABEL: @test_arraytail_braces()
// CHECK:       %braces = alloca %struct.arraytail, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(arraytail, arraytail, { 0xdead });
// CHECK-LABEL: @test_arraytail_custom()
// CHECK:       %custom = alloca %struct.arraytail, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(int0, int[0]);
// CHECK-LABEL: @test_int0_uninit()
// CHECK:       %uninit = alloca [0 x i32], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int0, int[0]);
// CHECK-LABEL: @test_int0_braces()
// CHECK:       %braces = alloca [0 x i32], align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 0, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(int1, int[1]);
// CHECK-LABEL: @test_int1_uninit()
// CHECK:       %uninit = alloca [1 x i32], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int1, int[1]);
// CHECK-LABEL: @test_int1_braces()
// CHECK:       %braces = alloca [1 x i32], align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(int1, int[1], { 0x33333333 });
// CHECK-LABEL: @test_int1_custom()
// CHECK:       %custom = alloca [1 x i32], align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(int64, int[64]);
// CHECK-LABEL: @test_int64_uninit()
// CHECK:       %uninit = alloca [64 x i32], align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(int64, int[64]);
// CHECK-LABEL: @test_int64_braces()
// CHECK:       %braces = alloca [64 x i32], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 16 %{{.*}}, i8 0, i64 256, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(int64, int[64], = { 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111, 0x11111111 });
// CHECK-LABEL: @test_int64_custom()
// CHECK:       %custom = alloca [64 x i32], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 16 %{{.*}}, i8 17, i64 256, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(bool4, bool[4]);
// CHECK-LABEL: @test_bool4_uninit()
// CHECK:       %uninit = alloca [4 x i8], align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(bool4, bool[4]);
// CHECK-LABEL: @test_bool4_braces()
// CHECK:       %braces = alloca [4 x i8], align 1
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 1 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(bool4, bool[4], { true, true, true, true });
// CHECK-LABEL: @test_bool4_custom()
// CHECK:       %custom = alloca [4 x i8], align 1
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(intptr4, int*[4]);
// CHECK-LABEL: @test_intptr4_uninit()
// CHECK:       %uninit = alloca [4 x i32*], align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(intptr4, int*[4]);
// CHECK-LABEL: @test_intptr4_braces()
// CHECK:       %braces = alloca [4 x i32*], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 16 %{{.*}}, i8 0, i64 32, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

  TEST_CUSTOM(intptr4, int*[4], = { (int*)0x22222222, (int*)0x22222222, (int*)0x22222222, (int*)0x22222222 });
// CHECK-LABEL: @test_intptr4_custom()
// CHECK:       %custom = alloca [4 x i32*], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(tailpad4, tailpad[4]);
// CHECK-LABEL: @test_tailpad4_uninit()
// CHECK:       %uninit = alloca [4 x %struct.tailpad], align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(tailpad4, tailpad[4]);
// CHECK-LABEL: @test_tailpad4_braces()
// CHECK:       %braces = alloca [4 x %struct.tailpad], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 16 %{{.*}}, i8 0, i64 16, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(tailpad4, tailpad[4], { {17, 1}, {17, 1}, {17, 1}, {17, 1} });
// CHECK-LABEL: @test_tailpad4_custom()
// CHECK:       %custom = alloca [4 x %struct.tailpad], align 16
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(atomicbool, _Atomic(bool));
// CHECK-LABEL: @test_atomicbool_uninit()
// CHECK:       %uninit = alloca i8, align 1
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_UNINIT(atomicint, _Atomic(int));
// CHECK-LABEL: @test_atomicint_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_UNINIT(atomicdouble, _Atomic(double));
// CHECK-LABEL: @test_atomicdouble_uninit()
// CHECK:       %uninit = alloca double, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_UNINIT(atomicnotlockfree, _Atomic(notlockfree));
// CHECK-LABEL: @test_atomicnotlockfree_uninit()
// CHECK:       %uninit = alloca %struct.notlockfree, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_UNINIT(atomicpadded, _Atomic(padded));
// CHECK-LABEL: @test_atomicpadded_uninit()
// CHECK:       %uninit = alloca %struct.padded, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_UNINIT(atomictailpad, _Atomic(tailpad));
// CHECK-LABEL: @test_atomictailpad_uninit()
// CHECK:       %uninit = alloca %struct.tailpad, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)


TEST_UNINIT(complexfloat, _Complex float);
// CHECK-LABEL: @test_complexfloat_uninit()
// CHECK:       %uninit = alloca { float, float }, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(complexfloat, _Complex float);
// CHECK-LABEL: @test_complexfloat_braces()
// CHECK:       %braces = alloca { float, float }, align 4
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %braces, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %braces, i32 0, i32 1
// CHECK-NEXT:  store float 0.000000e+00, float* %[[R]], align 4
// CHECK-NEXT:  store float 0.000000e+00, float* %[[I]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(complexfloat, _Complex float, { 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_complexfloat_custom()
// CHECK:       %custom = alloca { float, float }, align 4
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %custom, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { float, float }, { float, float }* %custom, i32 0, i32 1
// CHECK-NEXT:  store float 0x400921FB60000000, float* %[[R]], align 4
// CHECK-NEXT:  store float 0x400921FB60000000, float* %[[I]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(complexdouble, _Complex double);
// CHECK-LABEL: @test_complexdouble_uninit()
// CHECK:       %uninit = alloca { double, double }, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(complexdouble, _Complex double);
// CHECK-LABEL: @test_complexdouble_braces()
// CHECK:       %braces = alloca { double, double }, align 8
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %braces, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %braces, i32 0, i32 1
// CHECK-NEXT:  store double 0.000000e+00, double* %[[R]], align 8
// CHECK-NEXT:  store double 0.000000e+00, double* %[[I]], align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(complexdouble, _Complex double, { 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_complexdouble_custom()
// CHECK:       %custom = alloca { double, double }, align 8
// CHECK-NEXT:  %[[R:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %custom, i32 0, i32 0
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds { double, double }, { double, double }* %custom, i32 0, i32 1
// CHECK-NEXT:  store double 0x400921FB54442D18, double* %[[R]], align 8
// CHECK-NEXT:  store double 0x400921FB54442D18, double* %[[I]], align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(volatileint, volatile int);
// CHECK-LABEL: @test_volatileint_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(volatileint, volatile int);
// CHECK-LABEL: @test_volatileint_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store volatile i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(semivolatile, semivolatile);
// CHECK-LABEL: @test_semivolatile_uninit()
// CHECK:       %uninit = alloca %struct.semivolatile, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(semivolatile, semivolatile);
// CHECK-LABEL: @test_semivolatile_braces()
// CHECK:       %braces = alloca %struct.semivolatile, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(semivolatile, semivolatile, { 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_semivolatile_custom()
// CHECK:       %custom = alloca %struct.semivolatile, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(semivolatileinit, semivolatileinit);
// CHECK-LABEL: @test_semivolatileinit_uninit()
// CHECK:       %uninit = alloca %struct.semivolatileinit, align 4
// CHECK-NEXT:  call void @{{.*}}semivolatileinit{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(semivolatileinit, semivolatileinit);
// CHECK-LABEL: @test_semivolatileinit_braces()
// CHECK:       %braces = alloca %struct.semivolatileinit, align 4
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %braces, i32 0, i32 0
// CHECK-NEXT:  store i32 286331153, i32* %[[I]], align 4
// CHECK-NEXT:  %[[VI:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %braces, i32 0, i32 1
// CHECK-NEXT:  store volatile i32 286331153, i32* %[[VI]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(semivolatileinit, semivolatileinit, { 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_semivolatileinit_custom()
// CHECK:       %custom = alloca %struct.semivolatileinit, align 4
// CHECK-NEXT:  %[[I:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %custom, i32 0, i32 0
// CHECK-NEXT:  store i32 1145324612, i32* %[[I]], align 4
// CHECK-NEXT:  %[[VI:[^ ]*]] = getelementptr inbounds %struct.semivolatileinit, %struct.semivolatileinit* %custom, i32 0, i32 1
// CHECK-NEXT:  store volatile i32 1145324612, i32* %[[VI]], align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(base, base);
// CHECK-LABEL: @test_base_uninit()
// CHECK:       %uninit = alloca %struct.base, align 8
// CHECK-NEXT:  call void @{{.*}}base{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(base, base);
// CHECK-LABEL: @test_base_braces()
// CHECK:       %braces = alloca %struct.base, align 8
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 8 %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NEXT:  call void @{{.*}}base{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(derived, derived);
// CHECK-LABEL: @test_derived_uninit()
// CHECK:       %uninit = alloca %struct.derived, align 8
// CHECK-NEXT:  call void @{{.*}}derived{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(derived, derived);
// CHECK-LABEL: @test_derived_braces()
// CHECK:       %braces = alloca %struct.derived, align 8
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 8 %{{.*}}, i8 0, i64 8, i1 false)
// CHECK-NEXT:  call void @{{.*}}derived{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_UNINIT(virtualderived, virtualderived);
// CHECK-LABEL: @test_virtualderived_uninit()
// CHECK:       %uninit = alloca %struct.virtualderived, align 8
// CHECK-NEXT:  call void @{{.*}}virtualderived{{.*}}%uninit)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(virtualderived, virtualderived);
// CHECK-LABEL: @test_virtualderived_braces()
// CHECK:       %braces = alloca %struct.virtualderived, align 8
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 8 %{{.*}}, i8 0, i64 16, i1 false)
// CHECK-NEXT:  call void @{{.*}}virtualderived{{.*}}%braces)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)


TEST_UNINIT(matching, matching);
// CHECK-LABEL: @test_matching_uninit()
// CHECK:       %uninit = alloca %union.matching, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(matching, matching);
// CHECK-LABEL: @test_matching_braces()
// CHECK:       %braces = alloca %union.matching, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(matching, matching, { .f = 0xf00f });
// CHECK-LABEL: @test_matching_custom()
// CHECK:       %custom = alloca %union.matching, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(matchingreverse, matchingreverse);
// CHECK-LABEL: @test_matchingreverse_uninit()
// CHECK:       %uninit = alloca %union.matchingreverse, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(matchingreverse, matchingreverse);
// CHECK-LABEL: @test_matchingreverse_braces()
// CHECK:       %braces = alloca %union.matchingreverse, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(matchingreverse, matchingreverse, { .i = 0xf00f });
// CHECK-LABEL: @test_matchingreverse_custom()
// CHECK:       %custom = alloca %union.matchingreverse, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(unmatched, unmatched);
// CHECK-LABEL: @test_unmatched_uninit()
// CHECK:       %uninit = alloca %union.unmatched, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(unmatched, unmatched);
// CHECK-LABEL: @test_unmatched_braces()
// CHECK:       %braces = alloca %union.unmatched, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatched, unmatched, { .i = 0x3badbeef });
// CHECK-LABEL: @test_unmatched_custom()
// CHECK:       %custom = alloca %union.unmatched, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(unmatchedreverse, unmatchedreverse);
// CHECK-LABEL: @test_unmatchedreverse_uninit()
// CHECK:       %uninit = alloca %union.unmatchedreverse, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(unmatchedreverse, unmatchedreverse);
// CHECK-LABEL: @test_unmatchedreverse_braces()
// CHECK:       %braces = alloca %union.unmatchedreverse, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memset{{.*}}(i8* align 4 %{{.*}}, i8 0, i64 4, i1 false)
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatchedreverse, unmatchedreverse, { .c = 42  });
// CHECK-LABEL: @test_unmatchedreverse_custom()
// CHECK:       %custom = alloca %union.unmatchedreverse, align 4
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(unmatchedfp, unmatchedfp);
// CHECK-LABEL: @test_unmatchedfp_uninit()
// CHECK:       %uninit = alloca %union.unmatchedfp, align 8
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(unmatchedfp, unmatchedfp);
// CHECK-LABEL: @test_unmatchedfp_braces()
// CHECK:       %braces = alloca %union.unmatchedfp, align 8
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(unmatchedfp, unmatchedfp, { .d = 3.1415926535897932384626433 });
// CHECK-LABEL: @test_unmatchedfp_custom()
// CHECK:       %custom = alloca %union.unmatchedfp, align 8
// CHECK-NEXT:  bitcast
// CHECK-NEXT:  call void @llvm.memcpy
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(emptyenum, emptyenum);
// CHECK-LABEL: @test_emptyenum_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(emptyenum, emptyenum);
// CHECK-LABEL: @test_emptyenum_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(emptyenum, emptyenum, { (emptyenum)42 });
// CHECK-LABEL: @test_emptyenum_custom()
// CHECK:       %custom = alloca i32, align 4
// CHECK-NEXT:  store i32 42, i32* %custom, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(smallenum, smallenum);
// CHECK-LABEL: @test_smallenum_uninit()
// CHECK:       %uninit = alloca i32, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(smallenum, smallenum);
// CHECK-LABEL: @test_smallenum_braces()
// CHECK:       %braces = alloca i32, align 4
// CHECK-NEXT:  store i32 0, i32* %braces, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(smallenum, smallenum, { (smallenum)42 });
// CHECK-LABEL: @test_smallenum_custom()
// CHECK:       %custom = alloca i32, align 4
// CHECK-NEXT:  store i32 42, i32* %custom, align 4
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


TEST_UNINIT(intvec16, int  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_intvec16_uninit()
// CHECK:       %uninit = alloca <4 x i32>, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(intvec16, int  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_intvec16_braces()
// CHECK:       %braces = alloca <4 x i32>, align 16
// CHECK-NEXT:  store <4 x i32> zeroinitializer, <4 x i32>* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

  TEST_CUSTOM(intvec16, int  __attribute__((vector_size(16))), { 0x44444444, 0x44444444, 0x44444444, 0x44444444 });
// CHECK-LABEL: @test_intvec16_custom()
// CHECK:       %custom = alloca <4 x i32>, align 16
// CHECK-NEXT:  store <4 x i32> <i32 1145324612, i32 1145324612, i32 1145324612, i32 1145324612>, <4 x i32>* %custom, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(longlongvec32, long long  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_longlongvec32_uninit()
// CHECK:       %uninit = alloca <4 x i64>, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(longlongvec32, long long  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_longlongvec32_braces()
// CHECK:       %braces = alloca <4 x i64>, align 16
// CHECK-NEXT:  store <4 x i64> zeroinitializer, <4 x i64>* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(longlongvec32, long long  __attribute__((vector_size(32))), { 0x3333333333333333, 0x3333333333333333, 0x3333333333333333, 0x3333333333333333 });
// CHECK-LABEL: @test_longlongvec32_custom()
// CHECK:       %custom = alloca <4 x i64>, align 16
// CHECK-NEXT:  store <4 x i64> <i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323, i64 3689348814741910323>, <4 x i64>* %custom, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(floatvec16, float  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_floatvec16_uninit()
// CHECK:       %uninit = alloca <4 x float>, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(floatvec16, float  __attribute__((vector_size(16))));
// CHECK-LABEL: @test_floatvec16_braces()
// CHECK:       %braces = alloca <4 x float>, align 16
// CHECK-NEXT:  store <4 x float> zeroinitializer, <4 x float>* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(floatvec16, float  __attribute__((vector_size(16))), { 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_floatvec16_custom()
// CHECK:       %custom = alloca <4 x float>, align 16
// CHECK-NEXT:  store <4 x float> <float 0x400921FB60000000, float 0x400921FB60000000, float 0x400921FB60000000, float 0x400921FB60000000>, <4 x float>* %custom, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)

TEST_UNINIT(doublevec32, double  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_doublevec32_uninit()
// CHECK:       %uninit = alloca <4 x double>, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%uninit)

TEST_BRACES(doublevec32, double  __attribute__((vector_size(32))));
// CHECK-LABEL: @test_doublevec32_braces()
// CHECK:       %braces = alloca <4 x double>, align 16
// CHECK-NEXT:  store <4 x double> zeroinitializer, <4 x double>* %braces, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%braces)

TEST_CUSTOM(doublevec32, double  __attribute__((vector_size(32))), { 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433, 3.1415926535897932384626433 });
// CHECK-LABEL: @test_doublevec32_custom()
// CHECK:       %custom = alloca <4 x double>, align 16
// CHECK-NEXT:  store <4 x double> <double 0x400921FB54442D18, double 0x400921FB54442D18, double 0x400921FB54442D18, double 0x400921FB54442D18>, <4 x double>* %custom, align 16
// CHECK-NEXT:  call void @{{.*}}used{{.*}}%custom)


} // extern "C"
