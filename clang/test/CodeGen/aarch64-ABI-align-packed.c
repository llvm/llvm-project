// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple aarch64 -target-feature +neon -emit-llvm -O2 -o - %s | FileCheck %s
#include <stdarg.h>
#include <arm_neon.h>

// natural alignment 16, adjusted alignment 16
// expected alignment of copy on callee stack: 16
struct non_packed_struct {
  uint16x8_t M0; // member alignment 16
};

// natural alignment 1, adjusted alignment 1
// expected alignment of copy on callee stack: 8
struct __attribute((packed)) packed_struct {
  uint16x8_t M0; // member alignment 1, because the field is packed when the struct is packed
};

// natural alignment 1, adjusted alignment 1
// expected alignment of copy on callee stack: 8
struct packed_member {
  uint16x8_t M0 __attribute((packed)); // member alignment 1
};

// natural alignment 16, adjusted alignment 16, despite the 8-byte alignment specified by the attribute, because the natural alignment
// for the vector type is 16 and the attribute cannot decrease the minimum required alignment to be less.
// expected alignment of copy on callee stack: 16
struct __attribute((aligned (8))) aligned_struct_8 {
  uint16x8_t M0; // member alignment 16
};

// natural alignment 16, adjusted alignment 16
// expected alignment of copy on callee stack: 16
struct aligned_member_8 {
  uint16x8_t M0 __attribute((aligned (8))); // member alignment 16, despite the 8-byte alignment specified by the attribute,
  // because the natural alignment for the vector type is 16 and the attribute cannot decrease the minimum required alignment to be less.
};

// natural alignment 8, adjusted alignment 8
// expected alignment of copy on callee stack: 8
#pragma pack(8)
struct pragma_packed_struct_8 {
  uint16x8_t M0; // member alignment 8 because the struct is subject to packed(8)
};

// natural alignment 4, adjusted alignment 4
// expected alignment of copy on callee stack: 8
#pragma pack(4)
struct pragma_packed_struct_4 {
  uint16x8_t M0; // member alignment 4 because the struct is subject to packed(4)
};

double gd;
void init(int, ...);

struct non_packed_struct gs_non_packed_struct;

// CHECK-LABEL: define dso_local void @named_arg_non_packed_struct
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(16) [[S_NON_PACKED_STRUCT_COERCE:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_NON_PACKED_STRUCT_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_NON_PACKED_STRUCT_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2:![0-9]+]]
// CHECK-NEXT:    store <8 x i16> [[S_NON_PACKED_STRUCT_COERCE_FCA_0_EXTRACT]], ptr @gs_non_packed_struct, align 16, !tbaa [[TBAA6:![0-9]+]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_non_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct non_packed_struct s_non_packed_struct) {
    gd = d8;
    gs_non_packed_struct = s_non_packed_struct;
}

// CHECK-LABEL: define dso_local void @variadic_non_packed_struct
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6:[0-9]+]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_non_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct non_packed_struct on_callee_stack;
  on_callee_stack = va_arg(vl, struct non_packed_struct);
}

// CHECK-LABEL: define dso_local void @test_non_packed_struct
// CHECK-SAME: () local_unnamed_addr #[[ATTR4:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_NON_PACKED_STRUCT:%.*]] = alloca [[STRUCT_NON_PACKED_STRUCT:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_NON_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_NON_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_NON_PACKED_STRUCT]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_non_packed_struct(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_NON_PACKED_STRUCT]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_non_packed_struct(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_NON_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_non_packed_struct() {
    struct non_packed_struct s_non_packed_struct;
    init(1, &s_non_packed_struct);

    named_arg_non_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_non_packed_struct);
    variadic_non_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_non_packed_struct);
}

struct packed_struct gs_packed_struct;

// CHECK-LABEL: define dso_local void @named_arg_packed_struct
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(8) [[S_PACKED_STRUCT_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PACKED_STRUCT_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_PACKED_STRUCT_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_PACKED_STRUCT_COERCE_FCA_0_EXTRACT]], ptr @gs_packed_struct, align 1, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct packed_struct s_packed_struct) {
    gd = d8;
    gs_packed_struct = s_packed_struct;
}

// CHECK-LABEL: define dso_local void @variadic_packed_struct
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_packed_struct(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct packed_struct on_callee_stack;
  on_callee_stack = va_arg(vl, struct packed_struct);
}

// CHECK-LABEL: define dso_local void @test_packed_struct
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PACKED_STRUCT:%.*]] = alloca [[STRUCT_PACKED_STRUCT:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_PACKED_STRUCT]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_packed_struct(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_PACKED_STRUCT]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_packed_struct(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_PACKED_STRUCT]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_packed_struct() {
    struct packed_struct s_packed_struct;
    init(1, &s_packed_struct);

    named_arg_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_struct);
    variadic_packed_struct(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_struct);
}

struct packed_member gs_packed_member;

// CHECK-LABEL: define dso_local void @named_arg_packed_member
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(8) [[S_PACKED_MEMBER_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PACKED_MEMBER_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_PACKED_MEMBER_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_PACKED_MEMBER_COERCE_FCA_0_EXTRACT]], ptr @gs_packed_member, align 1, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_packed_member(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct packed_member s_packed_member) {
    gd = d8;
    gs_packed_member = s_packed_member;
}

// CHECK-LABEL: define dso_local void @variadic_packed_member
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_packed_member(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct packed_member on_callee_stack;
  on_callee_stack = va_arg(vl, struct packed_member);
}

// CHECK-LABEL: define dso_local void @test_packed_member
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PACKED_MEMBER:%.*]] = alloca [[STRUCT_PACKED_MEMBER:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_PACKED_MEMBER]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_PACKED_MEMBER]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_PACKED_MEMBER]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_packed_member(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_PACKED_MEMBER]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_packed_member(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_PACKED_MEMBER]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_packed_member() {
    struct packed_member s_packed_member;
    init(1, &s_packed_member);

    named_arg_packed_member(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_member);
    variadic_packed_member(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_packed_member);
}

struct aligned_struct_8 gs_aligned_struct_8;

// CHECK-LABEL: define dso_local void @named_arg_aligned_struct_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(16) [[S_ALIGNED_STRUCT_8_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_ALIGNED_STRUCT_8_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_ALIGNED_STRUCT_8_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_ALIGNED_STRUCT_8_COERCE_FCA_0_EXTRACT]], ptr @gs_aligned_struct_8, align 16, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_aligned_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct aligned_struct_8 s_aligned_struct_8) {
    gd = d8;
    gs_aligned_struct_8 = s_aligned_struct_8;
}

// CHECK-LABEL: define dso_local void @variadic_aligned_struct_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_aligned_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct aligned_struct_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct aligned_struct_8);
}

// CHECK-LABEL: define dso_local void @test_aligned_struct_8
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_ALIGNED_STRUCT_8:%.*]] = alloca [[STRUCT_ALIGNED_STRUCT_8:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_ALIGNED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_ALIGNED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_ALIGNED_STRUCT_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_aligned_struct_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_ALIGNED_STRUCT_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_aligned_struct_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_ALIGNED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_aligned_struct_8() {
    struct aligned_struct_8 s_aligned_struct_8;
    init(1, &s_aligned_struct_8);

    named_arg_aligned_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_struct_8);
    variadic_aligned_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_struct_8);
}

struct aligned_member_8 gs_aligned_member_8;

// CHECK-LABEL: define dso_local void @named_arg_aligned_member_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(16) [[S_ALIGNED_MEMBER_8_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_ALIGNED_MEMBER_8_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_ALIGNED_MEMBER_8_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_ALIGNED_MEMBER_8_COERCE_FCA_0_EXTRACT]], ptr @gs_aligned_member_8, align 16, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_aligned_member_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct aligned_member_8 s_aligned_member_8) {
    gd = d8;
    gs_aligned_member_8 = s_aligned_member_8;
}

// CHECK-LABEL: define dso_local void @variadic_aligned_member_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_aligned_member_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct aligned_member_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct aligned_member_8);
}

// CHECK-LABEL: define dso_local void @test_aligned_member_8
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_ALIGNED_MEMBER_8:%.*]] = alloca [[STRUCT_ALIGNED_MEMBER_8:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_ALIGNED_MEMBER_8]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_ALIGNED_MEMBER_8]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_ALIGNED_MEMBER_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_aligned_member_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_ALIGNED_MEMBER_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_aligned_member_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(16) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_ALIGNED_MEMBER_8]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_aligned_member_8() {
    struct aligned_member_8 s_aligned_member_8;
    init(1, &s_aligned_member_8);

    named_arg_aligned_member_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_member_8);
    variadic_aligned_member_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_aligned_member_8);
}

struct pragma_packed_struct_8 gs_pragma_packed_struct_8;

// CHECK-LABEL: define dso_local void @named_arg_pragma_packed_struct_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(8) [[S_PRAGMA_PACKED_STRUCT_8_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PRAGMA_PACKED_STRUCT_8_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_PRAGMA_PACKED_STRUCT_8_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_PRAGMA_PACKED_STRUCT_8_COERCE_FCA_0_EXTRACT]], ptr @gs_pragma_packed_struct_8, align 8, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_pragma_packed_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct pragma_packed_struct_8 s_pragma_packed_struct_8) {
    gd = d8;
    gs_pragma_packed_struct_8 = s_pragma_packed_struct_8;
}

// CHECK-LABEL: define dso_local void @variadic_pragma_packed_struct_8
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_pragma_packed_struct_8(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct pragma_packed_struct_8 on_callee_stack;
  on_callee_stack = va_arg(vl, struct pragma_packed_struct_8);
}

// CHECK-LABEL: define dso_local void @test_pragma_packed_struct_8
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PRAGMA_PACKED_STRUCT_8:%.*]] = alloca [[STRUCT_PRAGMA_PACKED_STRUCT_8:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_PRAGMA_PACKED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_PRAGMA_PACKED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_PRAGMA_PACKED_STRUCT_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_pragma_packed_struct_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_PRAGMA_PACKED_STRUCT_8]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_pragma_packed_struct_8(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_PRAGMA_PACKED_STRUCT_8]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_pragma_packed_struct_8() {
    struct pragma_packed_struct_8 s_pragma_packed_struct_8;
    init(1, &s_pragma_packed_struct_8);

    named_arg_pragma_packed_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_8);
    variadic_pragma_packed_struct_8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_8);
}

struct pragma_packed_struct_4 gs_pragma_packed_struct_4;

// CHECK-LABEL: define dso_local void @named_arg_pragma_packed_struct_4
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double noundef [[D8:%.*]], [1 x <8 x i16>] alignstack(8) [[S_PRAGMA_PACKED_STRUCT_4_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PRAGMA_PACKED_STRUCT_4_COERCE_FCA_0_EXTRACT:%.*]] = extractvalue [1 x <8 x i16>] [[S_PRAGMA_PACKED_STRUCT_4_COERCE]], 0
// CHECK-NEXT:    store double [[D8]], ptr @gd, align 8, !tbaa [[TBAA2]]
// CHECK-NEXT:    store <8 x i16> [[S_PRAGMA_PACKED_STRUCT_4_COERCE_FCA_0_EXTRACT]], ptr @gs_pragma_packed_struct_4, align 4, !tbaa [[TBAA6]]
// CHECK-NEXT:    ret void
__attribute__((noinline)) void named_arg_pragma_packed_struct_4(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, struct pragma_packed_struct_4 s_pragma_packed_struct_4) {
    gd = d8;
    gs_pragma_packed_struct_4 = s_pragma_packed_struct_4;
}

// CHECK-LABEL: define dso_local void @variadic_pragma_packed_struct_4
// CHECK-SAME: (double [[D0:%.*]], double [[D1:%.*]], double [[D2:%.*]], double [[D3:%.*]], double [[D4:%.*]], double [[D5:%.*]], double [[D6:%.*]], double [[D7:%.*]], double [[D8:%.*]], ...) local_unnamed_addr #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VL:%.*]] = alloca [[STRUCT___VA_LIST:%.*]], align 8
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    call void @llvm.va_start(ptr nonnull [[VL]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 32, ptr nonnull [[VL]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void variadic_pragma_packed_struct_4(double d0, double d1, double d2, double d3,
                                 double d4, double d5, double d6, double d7,
                                 double d8, ...) {
  va_list vl;
  va_start(vl, d8);
  struct pragma_packed_struct_4 on_callee_stack;
  on_callee_stack = va_arg(vl, struct pragma_packed_struct_4);
}

// CHECK-LABEL: define dso_local void @test_pragma_packed_struct_4
// CHECK-SAME: () local_unnamed_addr #[[ATTR4]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[S_PRAGMA_PACKED_STRUCT_4:%.*]] = alloca [[STRUCT_PRAGMA_PACKED_STRUCT_4:%.*]], align 16
// CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 16, ptr nonnull [[S_PRAGMA_PACKED_STRUCT_4]]) #[[ATTR6]]
// CHECK-NEXT:    call void (i32, ...) @init(i32 noundef 1, ptr noundef nonnull [[S_PRAGMA_PACKED_STRUCT_4]]) #[[ATTR6]]
// CHECK-NEXT:    [[DOTFCA_0_LOAD:%.*]] = load <8 x i16>, ptr [[S_PRAGMA_PACKED_STRUCT_4]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD]], 0
// CHECK-NEXT:    call void @named_arg_pragma_packed_struct_4(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double noundef 2.000000e+00, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT]])
// CHECK-NEXT:    [[DOTFCA_0_LOAD3:%.*]] = load <8 x i16>, ptr [[S_PRAGMA_PACKED_STRUCT_4]], align 16
// CHECK-NEXT:    [[DOTFCA_0_INSERT4:%.*]] = insertvalue [1 x <8 x i16>] poison, <8 x i16> [[DOTFCA_0_LOAD3]], 0
// CHECK-NEXT:    call void (double, double, double, double, double, double, double, double, double, ...) @variadic_pragma_packed_struct_4(double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, double poison, [1 x <8 x i16>] alignstack(8) [[DOTFCA_0_INSERT4]])
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 16, ptr nonnull [[S_PRAGMA_PACKED_STRUCT_4]]) #[[ATTR6]]
// CHECK-NEXT:    ret void
void test_pragma_packed_struct_4() {
    struct pragma_packed_struct_4 s_pragma_packed_struct_4;
    init(1, &s_pragma_packed_struct_4);

    named_arg_pragma_packed_struct_4(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_4);
    variadic_pragma_packed_struct_4(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, s_pragma_packed_struct_4);
}
//.
// CHECK: [[TBAA2]] = !{[[META3:![0-9]+]], [[META3]], i64 0}
// CHECK: [[META3]] = !{!"double", [[META4:![0-9]+]], i64 0}
// CHECK: [[META4]] = !{!"omnipotent char", [[META5:![0-9]+]], i64 0}
// CHECK: [[META5]] = !{!"Simple C/C++ TBAA"}
// CHECK: [[TBAA6]] = !{[[META4]], [[META4]], i64 0}
//.
