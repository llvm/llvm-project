// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-feature +neon -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=APCS-GNU %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-feature +neon -target-abi aapcs -emit-llvm -w -o - %s | FileCheck -check-prefix=AAPCS %s

// APCS-GNU-LABEL: define{{.*}} signext i8 @f0()
// AAPCS-LABEL: define{{.*}} arm_aapcscc signext i8 @f0()
char f0(void) {
  return 0;
}

// APCS-GNU-LABEL: define{{.*}} i8 @f1()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i8 @f1()
struct s1 { char f0; };
struct s1 f1(void) {}

// APCS-GNU-LABEL: define{{.*}} i16 @f2()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i16 @f2()
struct s2 { short f0; };
struct s2 f2(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f3()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f3()
struct s3 { int f0; };
struct s3 f3(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f4()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f4()
struct s4 { struct s4_0 { int f0; } f0; };
struct s4 f4(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f5(
// APCS-GNU: struct.s5* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f5()
struct s5 { struct { } f0; int f1; };
struct s5 f5(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f6(
// APCS-GNU: struct.s6* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f6()
struct s6 { int f0[1]; };
struct s6 f6(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f7()
// AAPCS-LABEL: define{{.*}} arm_aapcscc void @f7()
struct s7 { struct { int : 0; } f0; };
struct s7 f7(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f8(
// APCS-GNU: struct.s8* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc void @f8()
struct s8 { struct { int : 0; } f0[1]; };
struct s8 f8(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f9()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f9()
struct s9 { int f0; int : 0; };
struct s9 f9(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f10()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f10()
struct s10 { int f0; int : 0; int : 0; };
struct s10 f10(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f11(
// APCS-GNU: struct.s11* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f11()
struct s11 { int : 0; int f0; };
struct s11 f11(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f12()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f12()
union u12 { char f0; short f1; int f2; };
union u12 f12(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f13(
// APCS-GNU: struct.s13* noalias sret

// FIXME: This should return a float.
// AAPCS-FIXME: darm_aapcscc efine float @f13()
struct s13 { float f0; };
struct s13 f13(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f14(
// APCS-GNU: union.u14* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f14()
union u14 { float f0; };
union u14 f14(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f15()
// AAPCS-LABEL: define{{.*}} arm_aapcscc void @f15()
void f15(struct s7 a0) {}

// APCS-GNU-LABEL: define{{.*}} void @f16()
// AAPCS-LABEL: define{{.*}} arm_aapcscc void @f16()
void f16(struct s8 a0) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f17()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f17()
struct s17 { short f0 : 13; char f1 : 4; };
struct s17 f17(void) {}

// APCS-GNU-LABEL: define{{.*}} i32 @f18()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f18()
struct s18 { short f0; char f1 : 4; };
struct s18 f18(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f19(
// APCS-GNU: struct.s19* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f19()
struct s19 { int f0; struct s8 f1; };
struct s19 f19(void) {}

// APCS-GNU-LABEL: define{{.*}} void @f20(
// APCS-GNU: struct.s20* noalias sret
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f20()
struct s20 { struct s8 f1; int f0; };
struct s20 f20(void) {}

// APCS-GNU-LABEL: define{{.*}} i8 @f21()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f21()
struct s21 { struct {} f1; int f0 : 4; };
struct s21 f21(void) {}

// APCS-GNU-LABEL: define{{.*}} i16 @f22()
// APCS-GNU-LABEL: define{{.*}} i32 @f23()
// APCS-GNU-LABEL: define{{.*}} i64 @f24()
// APCS-GNU-LABEL: define{{.*}} i128 @f25()
// APCS-GNU-LABEL: define{{.*}} i64 @f26()
// APCS-GNU-LABEL: define{{.*}} i128 @f27()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i16 @f22()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f23()
// AAPCS: define{{.*}} arm_aapcscc void @f24({{.*}} noalias sret
// AAPCS: define{{.*}} arm_aapcscc void @f25({{.*}} noalias sret
// AAPCS: define{{.*}} arm_aapcscc void @f26({{.*}} noalias sret
// AAPCS: define{{.*}} arm_aapcscc void @f27({{.*}} noalias sret
_Complex char       f22(void) {}
_Complex short      f23(void) {}
_Complex int        f24(void) {}
_Complex long long  f25(void) {}
_Complex float      f26(void) {}
_Complex double     f27(void) {}

// APCS-GNU-LABEL: define{{.*}} i16 @f28()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i16 @f28()
struct s28 { _Complex char f0; };
struct s28 f28() {}

// APCS-GNU-LABEL: define{{.*}} i32 @f29()
// AAPCS-LABEL: define{{.*}} arm_aapcscc i32 @f29()
struct s29 { _Complex short f0; };
struct s29 f29() {}

// APCS-GNU: define{{.*}} void @f30({{.*}} noalias sret
// AAPCS: define{{.*}} arm_aapcscc void @f30({{.*}} noalias sret
struct s30 { _Complex int f0; };
struct s30 f30() {}

// PR11905
struct s31 { char x; };
void f31(struct s31 s) { }
// AAPCS: @f31([1 x i32] %s.coerce)
// AAPCS: %s = alloca %struct.s31, align 1
// AAPCS: [[TEMP:%.*]] = alloca [1 x i32], align 4
// AAPCS: store [1 x i32] %s.coerce, [1 x i32]* [[TEMP]], align 4
// APCS-GNU: @f31([1 x i32] %s.coerce)
// APCS-GNU: %s = alloca %struct.s31, align 1
// APCS-GNU: [[TEMP:%.*]] = alloca [1 x i32], align 4
// APCS-GNU: store [1 x i32] %s.coerce, [1 x i32]* [[TEMP]], align 4

// PR13562
struct s32 { double x; };
void f32(struct s32 s) { }
// AAPCS: @f32([1 x i64] %s.coerce)
// APCS-GNU: @f32([2 x i32] %s.coerce)

// PR13350
struct s33 { char buf[32*32]; };
void f33(struct s33 s) { }
// APCS-GNU-LABEL: define{{.*}} void @f33(%struct.s33* noundef byval(%struct.s33) align 4 %s)
// AAPCS-LABEL: define{{.*}} arm_aapcscc void @f33(%struct.s33* noundef byval(%struct.s33) align 4 %s)

// PR14048
struct s34 { char c; };
void f34(struct s34 s);
void g34(struct s34 *s) { f34(*s); }
// AAPCS: @g34(%struct.s34* noundef %s)
// AAPCS: %[[a:.*]] = alloca [1 x i32]
// AAPCS: load [1 x i32], [1 x i32]* %[[a]]

// rdar://12596507
struct s35
{
   float v[18]; //make sure byval is on.
} __attribute__((aligned(16)));
typedef struct s35 s35_with_align;

typedef __attribute__((neon_vector_type(4))) float float32x4_t;
static __attribute__((__always_inline__, __nodebug__)) float32x4_t vaddq_f32(
       float32x4_t __a, float32x4_t __b) {
 return __a + __b;
}
float32x4_t f35(int i, s35_with_align s1, s35_with_align s2) {
  float32x4_t v = vaddq_f32(*(float32x4_t *)&s1,
                            *(float32x4_t *)&s2);
  return v;
}
// APCS-GNU-LABEL: define{{.*}} <4 x float> @f35(i32 noundef %i, %struct.s35* noundef byval(%struct.s35) align 4 %0, %struct.s35* noundef byval(%struct.s35) align 4 %1)
// APCS-GNU: %[[a:.*]] = alloca %struct.s35, align 16
// APCS-GNU: %[[b:.*]] = bitcast %struct.s35* %[[a]] to i8*
// APCS-GNU: %[[c:.*]] = bitcast %struct.s35* %0 to i8*
// APCS-GNU: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align {{[0-9]+}} %[[b]], i8* align {{[0-9]+}} %[[c]]
// APCS-GNU: %[[d:.*]] = bitcast %struct.s35* %[[a]] to <4 x float>*
// APCS-GNU: load <4 x float>, <4 x float>* %[[d]], align 16

// AAPCS-LABEL: define{{.*}} arm_aapcscc <4 x float> @f35(i32 noundef %i, %struct.s35* noundef byval(%struct.s35) align 4 %s1, %struct.s35* noundef byval(%struct.s35) align 4 %s2)
// AAPCS: %[[a_addr:.*]] = alloca <4 x float>, align 16
// AAPCS: %[[b_addr:.*]] = alloca <4 x float>, align 16
// AAPCS: %[[p1:.*]] = bitcast %struct.s35* %s1 to <4 x float>*
// AAPCS: %[[a:.*]] = load <4 x float>, <4 x float>* %[[p1]], align 4
// AAPCS: %[[p2:.*]] = bitcast %struct.s35* %s2 to <4 x float>*
// AAPCS: %[[b:.*]] = load <4 x float>, <4 x float>* %[[p2]], align 4
// AAPCS: store <4 x float> %[[a]], <4 x float>* %[[a_addr]], align 16
// AAPCS: store <4 x float> %[[b]], <4 x float>* %[[b_addr]], align 16
