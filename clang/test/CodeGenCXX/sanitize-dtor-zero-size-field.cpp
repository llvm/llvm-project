// RUN: %clang_cc1 -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++20 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not "call void @__sanitizer_dtor_callback_fields"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++20 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not "call void @__sanitizer_dtor_callback_fields"

struct Empty {};

struct EmptyNonTrivial {
  ~EmptyNonTrivial();
};

struct Trivial {
  int a;
  char c;
};
static_assert(sizeof(Trivial) == 8);

struct NonTrivial {
  int a;
  char c;
  ~NonTrivial();
};
static_assert(sizeof(NonTrivial) == 8);

namespace T0 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  ~Struct(){};
} var;
static_assert(sizeof(Struct) == 16);
} // namespace T0
// CHECK-LABEL: define {{.*}} @_ZN2T06StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK-NEXT:    ret void

namespace empty {
namespace T1 {
struct Struct {
  NonTrivial nt;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T1
// CHECK-LABEL: define {{.*}} @_ZN5empty2T16StructD2Ev(
// CHECK:         [[GEP:%.+]] = getelementptr i8, {{.*}}, i64 8{{$}}
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr [[GEP]], i64 13)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T2 {
struct Struct {
  Trivial f1;
  NonTrivial nt;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T2
// CHECK-LABEL: define {{.*}} @_ZN5empty2T26StructD2Ev(
// CHECK:         [[GEP1:%.+]] = getelementptr i8, {{.*}}, i64 16{{$}}
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr [[GEP1]], i64 5)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         [[GEP2:%.+]] = getelementptr i8, {{.*}}, i64 0{{$}}
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr [[GEP2]], i64 8)
// CHECK-NEXT:    ret void

namespace T3 {
struct Struct {
  Trivial f1;
  int f2;
  NonTrivial nt;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T3
// CHECK-LABEL: define {{.*}} @_ZN5empty2T36StructD2Ev(
// CHECK:         [[GEP1:%.+]] = getelementptr i8, {{.*}}, i64 20{{$}}
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr [[GEP1]], i64 1)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         [[GEP2:%.+]] = getelementptr i8, {{.*}}, i64 0{{$}}
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr [[GEP2]], i64 12)
// CHECK-NEXT:    ret void

namespace T4 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  NonTrivial nt;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T4
// CHECK-LABEL: define {{.*}} @_ZN5empty2T46StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK-NEXT:    ret void

namespace T5 {
struct Struct {
  [[no_unique_address]] Empty e;
  NonTrivial nt;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T5
// CHECK-LABEL: define {{.*}} @_ZN5empty2T56StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T6 {
struct Struct {
  NonTrivial nt;
  [[no_unique_address]] Empty e;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T6
// CHECK-LABEL: define {{.*}} @_ZN5empty2T66StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T7 {
struct Struct {
  Trivial f1;
  NonTrivial nt;
  [[no_unique_address]] Empty e;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T7
// CHECK-LABEL: define {{.*}} @_ZN5empty2T76StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 5)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 8)
// CHECK-NEXT:    ret void

namespace T8 {
struct Struct {
  Trivial f1;
  [[no_unique_address]] Empty e;
  NonTrivial nt;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T8
// CHECK-LABEL: define {{.*}} @_ZN5empty2T86StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 5)
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 8)
// CHECK-NEXT:    ret void

namespace T9 {
struct Struct {
  Trivial f1;
  int f2;
  NonTrivial nt;
  [[no_unique_address]] Empty e;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T9
// CHECK-LABEL: define {{.*}} @_ZN5empty2T96StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 1)
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 12)
// CHECK-NEXT:    ret void

namespace T10 {
struct Struct {
  Trivial f1;
  int f2;
  [[no_unique_address]] Empty e;
  NonTrivial nt;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T10
// CHECK-LABEL: define {{.*}} @_ZN5empty3T106StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 1)
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 12)
// CHECK-NEXT:    ret void

namespace T11 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  NonTrivial nt;
  [[no_unique_address]] Empty e;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T11
// CHECK-LABEL: define {{.*}} @_ZN5empty3T116StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK-NEXT:    ret void

namespace T12 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  [[no_unique_address]] Empty e;
  NonTrivial nt;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T12
} // namespace empty
// CHECK-LABEL: define {{.*}} @_ZN5empty3T126StructD2Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK-NEXT:    ret void

namespace empty_non_trivial {
namespace T1 {
struct Struct {
  NonTrivial nt;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T1
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T16StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T2 {
struct Struct {
  Trivial f1;
  NonTrivial nt;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T2
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T26StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 5)
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 8)
// CHECK-NEXT:    ret void

namespace T3 {
struct Struct {
  Trivial f1;
  int f2;
  NonTrivial nt;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T3
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T36StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 1)
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 12)
// CHECK-NEXT:    ret void

namespace T4 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  NonTrivial nt;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T4
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T46StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK-NEXT:    ret void

namespace T5 {
struct Struct {
  [[no_unique_address]] EmptyNonTrivial e;
  NonTrivial nt;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T5
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T56StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T6 {
struct Struct {
  NonTrivial nt;
  [[no_unique_address]] EmptyNonTrivial e;
  Trivial f1;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T6
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T66StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 13)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T7 {
struct Struct {
  Trivial f1;
  NonTrivial nt;
  [[no_unique_address]] EmptyNonTrivial e;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T7
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T76StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 5)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 8)
// CHECK-NEXT:    ret void

namespace T8 {
struct Struct {
  Trivial f1;
  [[no_unique_address]] EmptyNonTrivial e;
  NonTrivial nt;
  int f2;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T8
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T86StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 5)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 8)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T9 {
struct Struct {
  Trivial f1;
  int f2;
  NonTrivial nt;
  [[no_unique_address]] EmptyNonTrivial e;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T9
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial2T96StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 1)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 12)
// CHECK-NEXT:    ret void

namespace T10 {
struct Struct {
  Trivial f1;
  int f2;
  [[no_unique_address]] EmptyNonTrivial e;
  NonTrivial nt;
  char f3;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T10
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial3T106StructD2Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 1)
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 12)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK-NEXT:    ret void

namespace T11 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  NonTrivial nt;
  [[no_unique_address]] EmptyNonTrivial e;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T11
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial3T116StructD2Ev(
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK-NEXT:    ret void

namespace T12 {
struct Struct {
  Trivial f1;
  int f2;
  char f3;
  [[no_unique_address]] EmptyNonTrivial e;
  NonTrivial nt;
} var;
static_assert(sizeof(Struct) == 24);
} // namespace T12
} // namespace empty_non_trivial
// CHECK-LABEL: define {{.*}} @_ZN17empty_non_trivial3T126StructD2Ev(
// CHECK:         call void @_ZN10NonTrivialD1Ev(
// CHECK:         call void @__sanitizer_dtor_callback_fields(ptr {{.*}}, i64 16)
// CHECK:         call void @_ZN15EmptyNonTrivialD1Ev(
// CHECK:         ret void
