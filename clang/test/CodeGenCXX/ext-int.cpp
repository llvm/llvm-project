// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,NoNewStructPathTBAA
// RUN: %clang_cc1 -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,NewStructPathTBAA

// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,NoNewStructPathTBAA
// RUN: %clang_cc1 -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,NewStructPathTBAA

#include <typeinfo>

// Ensure that the layout for these structs is the same as the normal bitfield
// layouts.
struct BitFieldsByte {
  _ExtInt(7) A : 3;
  _ExtInt(7) B : 3;
  _ExtInt(7) C : 2;
};
// CHECK: %struct.BitFieldsByte = type { i8 }

struct BitFieldsShort {
  _ExtInt(15) A : 3;
  _ExtInt(15) B : 3;
  _ExtInt(15) C : 2;
};
// LIN: %struct.BitFieldsShort = type { i8, i8 }
// WIN: %struct.BitFieldsShort = type { i16 }

struct BitFieldsInt {
  _ExtInt(31) A : 3;
  _ExtInt(31) B : 3;
  _ExtInt(31) C : 2;
};
// LIN: %struct.BitFieldsInt = type { i8, [3 x i8] }
// WIN: %struct.BitFieldsInt = type { i32 }

struct BitFieldsLong {
  _ExtInt(63) A : 3;
  _ExtInt(63) B : 3;
  _ExtInt(63) C : 2;
};
// LIN: %struct.BitFieldsLong = type { i8, [7 x i8] }
// WIN: %struct.BitFieldsLong = type { i64 }

struct HasExtIntFirst {
  _ExtInt(35) A;
  int B;
};
// CHECK: %struct.HasExtIntFirst = type { i35, i32 }

struct HasExtIntLast {
  int A;
  _ExtInt(35) B;
};
// CHECK: %struct.HasExtIntLast = type { i32, i35 }

struct HasExtIntMiddle {
  int A;
  _ExtInt(35) B;
  int C;
};
// CHECK: %struct.HasExtIntMiddle = type { i32, i35, i32 }

// Force emitting of the above structs.
void StructEmit() {
  BitFieldsByte A;
  BitFieldsShort B;
  BitFieldsInt C;
  BitFieldsLong D;

  HasExtIntFirst E;
  HasExtIntLast F;
  HasExtIntMiddle G;
}

void BitfieldAssignment() {
  // LIN: define{{.*}} void @_Z18BitfieldAssignmentv
  // WIN: define dso_local void  @"?BitfieldAssignment@@YAXXZ"
  BitFieldsByte B;
  B.A = 3;
  B.B = 2;
  B.C = 1;
  // First one is used for the lifetime start, skip that.
  // CHECK: bitcast %struct.BitFieldsByte*
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADA:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARA:.+]] = and i8 %[[LOADA]], -8
  // CHECK: %[[SETA:.+]] = or i8 %[[CLEARA]], 3
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADB:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARB:.+]] = and i8 %[[LOADB]], -57
  // CHECK: %[[SETB:.+]] = or i8 %[[CLEARB]], 16
  // CHECK: %[[BFType:.+]] = bitcast %struct.BitFieldsByte*
  // CHECK: %[[LOADC:.+]] = load i8, i8* %[[BFType]]
  // CHECK: %[[CLEARC:.+]] = and i8 %[[LOADC]], 63
  // CHECK: %[[SETC:.+]] = or i8 %[[CLEARC]], 64
}

enum AsEnumUnderlyingType : _ExtInt(9) {
  A,B,C
};

void UnderlyingTypeUsage(AsEnumUnderlyingType Param) {
  // LIN: define{{.*}} void @_Z19UnderlyingTypeUsage20AsEnumUnderlyingType(i9 signext %
  // WIN: define dso_local void @"?UnderlyingTypeUsage@@YAXW4AsEnumUnderlyingType@@@Z"(i9 %
  AsEnumUnderlyingType Var;
  // CHECK: alloca i9, align 2
  // CHECK: store i9 %{{.*}}, align 2
}

unsigned _ExtInt(33) ManglingTestRetParam(unsigned _ExtInt(33) Param) {
// LIN: define{{.*}} i64 @_Z20ManglingTestRetParamU7_ExtIntILi33EEj(i64 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_UExtInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

_ExtInt(33) ManglingTestRetParam(_ExtInt(33) Param) {
// LIN: define{{.*}} i64 @_Z20ManglingTestRetParamU7_ExtIntILi33EEi(i64 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_ExtInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

template<typename T>
void ManglingTestTemplateParam(T&);
template<_ExtInt(99) T>
void ManglingTestNTTP();

void ManglingInstantiator() {
  // LIN: define{{.*}} void @_Z20ManglingInstantiatorv()
  // WIN: define dso_local void @"?ManglingInstantiator@@YAXXZ"()
  _ExtInt(93) A;
  ManglingTestTemplateParam(A);
// LIN: call void @_Z25ManglingTestTemplateParamIU7_ExtIntILi93EEiEvRT_(i93*
// WIN: call void @"??$ManglingTestTemplateParam@U?$_ExtInt@$0FN@@__clang@@@@YAXAEAU?$_ExtInt@$0FN@@__clang@@@Z"(i93*
  constexpr _ExtInt(93) B = 993;
  ManglingTestNTTP<38>();
// LIN: call void @_Z16ManglingTestNTTPILU7_ExtIntILi99EEi38EEvv()
// WIN: call void @"??$ManglingTestNTTP@$0CG@@@YAXXZ"()
  ManglingTestNTTP<B>();
// LIN: call void @_Z16ManglingTestNTTPILU7_ExtIntILi99EEi993EEvv()
// WIN: call void @"??$ManglingTestNTTP@$0DOB@@@YAXXZ"()
}

void TakesVarargs(int i, ...) {
  // LIN: define{{.*}} void @_Z12TakesVarargsiz(i32 %i, ...)
  // WIN: define dso_local void @"?TakesVarargs@@YAXHZZ"(i32 %i, ...)

  __builtin_va_list args;
  // LIN: %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag]
  // WIN: %[[ARGS:.+]] = alloca i8*
  __builtin_va_start(args, i);
  // LIN: %[[STARTAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[STARTAD1:.+]] = bitcast %struct.__va_list_tag* %[[STARTAD]] to i8*
  // LIN: call void @llvm.va_start(i8* %[[STARTAD1]])
  // WIN: %[[ARGSLLIFETIMESTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: %[[ARGSSTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: call void @llvm.va_start(i8* %[[ARGSSTART]])

  _ExtInt(92) A = __builtin_va_arg(args, _ExtInt(92));
  // LIN: %[[AD1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[OFA_P1:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD1]], i32 0, i32 0
  // LIN: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P1]]
  // LIN: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 32
  // LIN: br i1 %[[FITSINGP]]
  // LIN: %[[BC1:.+]] = phi i92*
  // LIN: %[[LOAD1:.+]] = load i92, i92* %[[BC1]]
  // LIN: store i92 %[[LOAD1]], i92*

  // WIN: %[[CUR1:.+]] = load i8*, i8** %[[ARGS]]
  // WIN: %[[NEXT1:.+]] = getelementptr inbounds i8, i8* %[[CUR1]], i64 16
  // WIN: store i8* %[[NEXT1]], i8** %[[ARGS]]
  // WIN: %[[BC1:.+]] = bitcast i8* %[[CUR1]] to i92*
  // WIN: %[[LOADV1:.+]] = load i92, i92* %[[BC1]]
  // WIN: store i92 %[[LOADV1]], i92*

  _ExtInt(31) B = __builtin_va_arg(args, _ExtInt(31));
  // LIN: %[[AD2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[OFA_P2:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD2]], i32 0, i32 0
  // LIN: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P2]]
  // LIN: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN: br i1 %[[FITSINGP]]
  // LIN: %[[BC1:.+]] = phi i31*
  // LIN: %[[LOAD1:.+]] = load i31, i31* %[[BC1]]
  // LIN: store i31 %[[LOAD1]], i31*

  // WIN: %[[CUR2:.+]] = load i8*, i8** %[[ARGS]]
  // WIN: %[[NEXT2:.+]] = getelementptr inbounds i8, i8* %[[CUR2]], i64 8
  // WIN: store i8* %[[NEXT2]], i8** %[[ARGS]]
  // WIN: %[[BC2:.+]] = bitcast i8* %[[CUR2]] to i31*
  // WIN: %[[LOADV2:.+]] = load i31, i31* %[[BC2]]
  // WIN: store i31 %[[LOADV2]], i31*

  _ExtInt(16) C = __builtin_va_arg(args, _ExtInt(16));
  // LIN: %[[AD3:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[OFA_P3:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD3]], i32 0, i32 0
  // LIN: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P3]]
  // LIN: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN: br i1 %[[FITSINGP]]
  // LIN: %[[BC1:.+]] = phi i16*
  // LIN: %[[LOAD1:.+]] = load i16, i16* %[[BC1]]
  // LIN: store i16 %[[LOAD1]], i16*

  // WIN: %[[CUR3:.+]] = load i8*, i8** %[[ARGS]]
  // WIN: %[[NEXT3:.+]] = getelementptr inbounds i8, i8* %[[CUR3]], i64 8
  // WIN: store i8* %[[NEXT3]], i8** %[[ARGS]]
  // WIN: %[[BC3:.+]] = bitcast i8* %[[CUR3]] to i16*
  // WIN: %[[LOADV3:.+]] = load i16, i16* %[[BC3]]
  // WIN: store i16 %[[LOADV3]], i16*

  _ExtInt(129) D = __builtin_va_arg(args, _ExtInt(129));
  // LIN: %[[AD4:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[OFA_P4:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD4]], i32 0, i32 2
  // LIN: %[[OFA4:.+]] = load i8*, i8** %[[OFA_P4]]
  // LIN: %[[BC4:.+]] = bitcast i8* %[[OFA4]] to i129*
  // LIN: %[[OFANEXT4:.+]] = getelementptr i8, i8* %[[OFA4]], i32 24
  // LIN: store i8* %[[OFANEXT4]], i8** %[[OFA_P4]]
  // LIN: %[[LOAD4:.+]] = load i129, i129* %[[BC4]]
  // LIN: store i129 %[[LOAD4]], i129*

  // WIN: %[[CUR4:.+]] = load i8*, i8** %[[ARGS]]
  // WIN: %[[NEXT4:.+]] = getelementptr inbounds i8, i8* %[[CUR4]], i64 24
  // WIN: store i8* %[[NEXT4]], i8** %[[ARGS]]
  // WIN: %[[BC4:.+]] = bitcast i8* %[[CUR4]] to i129*
  // WIN: %[[LOADV4:.+]] = load i129, i129* %[[BC4]]
  // WIN: store i129 %[[LOADV4]], i129*

  _ExtInt(16777200) E = __builtin_va_arg(args, _ExtInt(16777200));
  // LIN: %[[AD5:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[OFA_P5:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD5]], i32 0, i32 2
  // LIN: %[[OFA5:.+]] = load i8*, i8** %[[OFA_P5]]
  // LIN: %[[BC5:.+]] = bitcast i8* %[[OFA5]] to i16777200*
  // LIN: %[[OFANEXT5:.+]] = getelementptr i8, i8* %[[OFA5]], i32 2097152
  // LIN: store i8* %[[OFANEXT5]], i8** %[[OFA_P5]]
  // LIN: %[[LOAD5:.+]] = load i16777200, i16777200* %[[BC5]]
  // LIN: store i16777200 %[[LOAD5]], i16777200*

  // WIN: %[[CUR5:.+]] = load i8*, i8** %[[ARGS]]
  // WIN: %[[NEXT5:.+]] = getelementptr inbounds i8, i8* %[[CUR5]], i64 2097152
  // WIN: store i8* %[[NEXT5]], i8** %[[ARGS]]
  // WIN: %[[BC5:.+]] = bitcast i8* %[[CUR5]] to i16777200*
  // WIN: %[[LOADV5:.+]] = load i16777200, i16777200* %[[BC5]]
  // WIN: store i16777200 %[[LOADV5]], i16777200*

  __builtin_va_end(args);
  // LIN: %[[ENDAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN: %[[ENDAD1:.+]] = bitcast %struct.__va_list_tag* %[[ENDAD]] to i8*
  // LIN: call void @llvm.va_end(i8* %[[ENDAD1]])
  // WIN: %[[ARGSEND:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: call void @llvm.va_end(i8* %[[ARGSEND]])
}
void typeid_tests() {
  // LIN: define{{.*}} void @_Z12typeid_testsv()
  // WIN: define dso_local void @"?typeid_tests@@YAXXZ"()
  unsigned _ExtInt(33) U33_1, U33_2;
  _ExtInt(33) S33_1, S33_2;
  _ExtInt(32) S32_1, S32_2;

 auto A = typeid(U33_1);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi33EEj to %"class.std::type_info"*))
 // WIN: call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UExtInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto B = typeid(U33_2);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi33EEj to %"class.std::type_info"*))
 // WIN:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UExtInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto C = typeid(S33_1);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi33EEi to %"class.std::type_info"*))
 // WIN:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_ExtInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto D = typeid(S33_2);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi33EEi to %"class.std::type_info"*))
 // WIN:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_ExtInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto E = typeid(S32_1);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi32EEi to %"class.std::type_info"*))
 // WIN:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_ExtInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
 auto F = typeid(S32_2);
 // LIN: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIU7_ExtIntILi32EEi to %"class.std::type_info"*))
 // WIN:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_ExtInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
}

void ExplicitCasts() {
  // LIN: define{{.*}} void @_Z13ExplicitCastsv()
  // WIN: define dso_local void @"?ExplicitCasts@@YAXXZ"()

  _ExtInt(33) a;
  _ExtInt(31) b;
  int i;

  a = i;
  // CHECK: %[[CONV:.+]] = sext i32 %{{.+}} to i33
  b = i;
  // CHECK: %[[CONV:.+]] = trunc i32 %{{.+}} to i31
  i = a;
  // CHECK: %[[CONV:.+]] = trunc i33 %{{.+}} to i32
  i = b;
  // CHECK: %[[CONV:.+]] = sext i31 %{{.+}} to i32
}

struct S {
  _ExtInt(17) A;
  _ExtInt(16777200) B;
  _ExtInt(17) C;
};

void OffsetOfTest() {
  // LIN: define{{.*}} void @_Z12OffsetOfTestv()
  // WIN: define dso_local void @"?OffsetOfTest@@YAXXZ"()

  auto A = __builtin_offsetof(S,A);
  // CHECK: store i64 0, i64* %{{.+}}
  auto B = __builtin_offsetof(S,B);
  // CHECK: store i64 8, i64* %{{.+}}
  auto C = __builtin_offsetof(S,C);
  // CHECK: store i64 2097160, i64* %{{.+}}
}


void ShiftExtIntByConstant(_ExtInt(28) Ext) {
// LIN: define{{.*}} void @_Z21ShiftExtIntByConstantU7_ExtIntILi28EEi
// WIN: define dso_local void @"?ShiftExtIntByConstant@@YAXU?$_ExtInt@$0BM@@__clang@@@Z"
  Ext << 7;
  // CHECK: shl i28 %{{.+}}, 7
  Ext >> 7;
  // CHECK: ashr i28 %{{.+}}, 7
  Ext << -7;
  // CHECK: shl i28 %{{.+}}, -7
  Ext >> -7;
  // CHECK: ashr i28 %{{.+}}, -7

  // UB in C/C++, Defined in OpenCL.
  Ext << 29;
  // CHECK: shl i28 %{{.+}}, 29
  Ext >> 29;
  // CHECK: ashr i28 %{{.+}}, 29
}

void ConstantShiftByExtInt(_ExtInt(28) Ext, _ExtInt(65) LargeExt) {
  // LIN: define{{.*}} void @_Z21ConstantShiftByExtIntU7_ExtIntILi28EEiU7_ExtIntILi65EEi
  // WIN: define dso_local void @"?ConstantShiftByExtInt@@YAXU?$_ExtInt@$0BM@@__clang@@U?$_ExtInt@$0EB@@2@@Z"
  10 << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: shl i32 10, %[[PROMO]]
  10 >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: ashr i32 10, %[[PROMO]]
  10 << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: shl i32 10, %[[PROMO]]
  10 >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: ashr i32 10, %[[PROMO]]
}

void Shift(_ExtInt(28) Ext, _ExtInt(65) LargeExt, int i) {
  // LIN: define{{.*}} void @_Z5ShiftU7_ExtIntILi28EEiU7_ExtIntILi65EEii
  // WIN: define dso_local void @"?Shift@@YAXU?$_ExtInt@$0BM@@__clang@@U?$_ExtInt@$0EB@@2@H@Z"
  i << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: shl i32 {{.+}}, %[[PROMO]]
  i >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i32
  // CHECK: ashr i32 {{.+}}, %[[PROMO]]

  i << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: shl i32 {{.+}}, %[[PROMO]]
  i >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i32
  // CHECK: ashr i32 {{.+}}, %[[PROMO]]

  Ext << i;
  // CHECK: %[[PROMO:.+]] = trunc i32 %{{.+}} to i28
  // CHECK: shl i28 {{.+}}, %[[PROMO]]
  Ext >> i;
  // CHECK: %[[PROMO:.+]] = trunc i32 %{{.+}} to i28
  // CHECK: ashr i28 {{.+}}, %[[PROMO]]

  LargeExt << i;
  // CHECK: %[[PROMO:.+]] = zext i32 %{{.+}} to i65
  // CHECK: shl i65 {{.+}}, %[[PROMO]]
  LargeExt >> i;
  // CHECK: %[[PROMO:.+]] = zext i32 %{{.+}} to i65
  // CHECK: ashr i65 {{.+}}, %[[PROMO]]

  Ext << LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i28
  // CHECK: shl i28 {{.+}}, %[[PROMO]]
  Ext >> LargeExt;
  // CHECK: %[[PROMO:.+]] = trunc i65 %{{.+}} to i28
  // CHECK: ashr i28 {{.+}}, %[[PROMO]]

  LargeExt << Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i65
  // CHECK: shl i65 {{.+}}, %[[PROMO]]
  LargeExt >> Ext;
  // CHECK: %[[PROMO:.+]] = zext i28 %{{.+}} to i65
  // CHECK: ashr i65 {{.+}}, %[[PROMO]]
}

void ComplexTest(_Complex _ExtInt(12) first,
                                 _Complex _ExtInt(33) second) {
  // LIN: define{{.*}} void @_Z11ComplexTestCU7_ExtIntILi12EEiCU7_ExtIntILi33EEi
  // WIN: define dso_local void  @"?ComplexTest@@YAXU?$_Complex@U?$_ExtInt@$0M@@__clang@@@__clang@@U?$_Complex@U?$_ExtInt@$0CB@@__clang@@@2@@Z"
  first + second;
  // CHECK: %[[FIRST_REALP:.+]] = getelementptr inbounds { i12, i12 }, { i12, i12 }* %{{.+}}, i32 0, i32 0
  // CHECK: %[[FIRST_REAL:.+]] = load i12, i12* %[[FIRST_REALP]]
  // CHECK: %[[FIRST_IMAGP:.+]] = getelementptr inbounds { i12, i12 }, { i12, i12 }* %{{.+}}, i32 0, i32 1
  // CHECK: %[[FIRST_IMAG:.+]] = load i12, i12* %[[FIRST_IMAGP]]
  // CHECK: %[[FIRST_REAL_CONV:.+]] = sext i12 %[[FIRST_REAL]]
  // CHECK: %[[FIRST_IMAG_CONV:.+]] = sext i12 %[[FIRST_IMAG]]
  // CHECK: %[[SECOND_REALP:.+]] = getelementptr inbounds { i33, i33 }, { i33, i33 }* %{{.+}}, i32 0, i32 0
  // CHECK: %[[SECOND_REAL:.+]] = load i33, i33* %[[SECOND_REALP]]
  // CHECK: %[[SECOND_IMAGP:.+]] = getelementptr inbounds { i33, i33 }, { i33, i33 }* %{{.+}}, i32 0, i32 1
  // CHECK: %[[SECOND_IMAG:.+]] = load i33, i33* %[[SECOND_IMAGP]]
  // CHECK: %[[REAL:.+]] = add i33 %[[FIRST_REAL_CONV]], %[[SECOND_REAL]]
  // CHECK: %[[IMAG:.+]] = add i33 %[[FIRST_IMAG_CONV]], %[[SECOND_IMAG]]
}

// Ensure that these types don't alias the normal int types.
void TBAATest(_ExtInt(sizeof(int) * 8) ExtInt,
              unsigned _ExtInt(sizeof(int) * 8) ExtUInt,
              _ExtInt(6) Other) {
  // CHECK-DAG: store i32 %{{.+}}, i32* %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA:.+]]
  // CHECK-DAG: store i32 %{{.+}}, i32* %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA]]
  // CHECK-DAG: store i6 %{{.+}}, i6* %{{.+}}, align 1, !tbaa ![[EXTINT6_TBAA:.+]]
  ExtInt = 5;
  ExtUInt = 5;
  Other = 5;
}

// NoNewStructPathTBAA-DAG: ![[CHAR_TBAA_ROOT:.+]] = !{!"omnipotent char", ![[TBAA_ROOT:.+]], i64 0}
// NoNewStructPathTBAA-DAG: ![[TBAA_ROOT]] = !{!"Simple C++ TBAA"}
// NoNewStructPathTBAA-DAG: ![[EXTINT_TBAA]] = !{![[EXTINT_TBAA_ROOT:.+]], ![[EXTINT_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{!"_ExtInt(32)", ![[CHAR_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{!"_ExtInt(6)", ![[CHAR_TBAA_ROOT]], i64 0}

// NewStructPathTBAA-DAG: ![[CHAR_TBAA_ROOT:.+]] = !{![[TBAA_ROOT:.+]], i64 1, !"omnipotent char"}
// NewStructPathTBAA-DAG: ![[TBAA_ROOT]] = !{!"Simple C++ TBAA"}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA]] = !{![[EXTINT_TBAA_ROOT:.+]], ![[EXTINT_TBAA_ROOT]], i64 0, i64 4}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 4, !"_ExtInt(32)"}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0, i64 1}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 1, !"_ExtInt(6)"}
