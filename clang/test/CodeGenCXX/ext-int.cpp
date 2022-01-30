// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN64,NoNewStructPathTBAA
// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN64,NewStructPathTBAA

// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN64,NoNewStructPathTBAA
// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN64,NewStructPathTBAA

// RUN: %clang_cc1 -disable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN32,NoNewStructPathTBAA
// RUN: %clang_cc1 -disable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN32,NewStructPathTBAA

// RUN: %clang_cc1 -disable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN32,NoNewStructPathTBAA
// RUN: %clang_cc1 -disable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN32,NewStructPathTBAA

namespace std {
  class type_info { public: virtual ~type_info(); private: const char * name; };
} // namespace std

// Ensure that the layout for these structs is the same as the normal bitfield
// layouts.
struct BitFieldsByte {
  _BitInt(7) A : 3;
  _BitInt(7) B : 3;
  _BitInt(7) C : 2;
};
// CHECK: %struct.BitFieldsByte = type { i8 }

struct BitFieldsShort {
  _BitInt(15) A : 3;
  _BitInt(15) B : 3;
  _BitInt(15) C : 2;
};
// LIN: %struct.BitFieldsShort = type { i8, i8 }
// WIN: %struct.BitFieldsShort = type { i16 }

struct BitFieldsInt {
  _BitInt(31) A : 3;
  _BitInt(31) B : 3;
  _BitInt(31) C : 2;
};
// LIN: %struct.BitFieldsInt = type { i8, [3 x i8] }
// WIN: %struct.BitFieldsInt = type { i32 }

struct BitFieldsLong {
  _BitInt(63) A : 3;
  _BitInt(63) B : 3;
  _BitInt(63) C : 2;
};
// LIN64: %struct.BitFieldsLong = type { i8, [7 x i8] }
// LIN32: %struct.BitFieldsLong = type { i8, [3 x i8] }
// WIN: %struct.BitFieldsLong = type { i64 }

struct HasBitIntFirst {
  _BitInt(35) A;
  int B;
};
// CHECK: %struct.HasBitIntFirst = type { i35, i32 }

struct HasBitIntLast {
  int A;
  _BitInt(35) B;
};
// CHECK: %struct.HasBitIntLast = type { i32, i35 }

struct HasBitIntMiddle {
  int A;
  _BitInt(35) B;
  int C;
};
// CHECK: %struct.HasBitIntMiddle = type { i32, i35, i32 }

// Force emitting of the above structs.
void StructEmit() {
  BitFieldsByte A;
  BitFieldsShort B;
  BitFieldsInt C;
  BitFieldsLong D;

  HasBitIntFirst E;
  HasBitIntLast F;
  HasBitIntMiddle G;
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

enum AsEnumUnderlyingType : _BitInt(9) {
  A,B,C
};

void UnderlyingTypeUsage(AsEnumUnderlyingType Param) {
  // LIN: define{{.*}} void @_Z19UnderlyingTypeUsage20AsEnumUnderlyingType(i9 signext %
  // WIN64: define dso_local void @"?UnderlyingTypeUsage@@YAXW4AsEnumUnderlyingType@@@Z"(i9 %
  // WIN32: define dso_local void @"?UnderlyingTypeUsage@@YAXW4AsEnumUnderlyingType@@@Z"(i9 signext %
  AsEnumUnderlyingType Var;
  // CHECK: alloca i9, align 2
  // CHECK: store i9 %{{.*}}, align 2
}

unsigned _BitInt(33) ManglingTestRetParam(unsigned _BitInt(33) Param) {
// LIN64: define{{.*}} i64 @_Z20ManglingTestRetParamDU33_(i64 %
// LIN32: define{{.*}} i33 @_Z20ManglingTestRetParamDU33_(i33 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_UBitInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

_BitInt(33) ManglingTestRetParam(_BitInt(33) Param) {
// LIN64: define{{.*}} i64 @_Z20ManglingTestRetParamDB33_(i64 %
// LIN32: define{{.*}} i33 @_Z20ManglingTestRetParamDB33_(i33 %
// WIN: define dso_local i33 @"?ManglingTestRetParam@@YAU?$_BitInt@$0CB@@__clang@@U12@@Z"(i33
  return 0;
}

template<typename T>
void ManglingTestTemplateParam(T&);
template<_BitInt(99) T>
void ManglingTestNTTP();
template <int N>
auto ManglingDependent() -> decltype(_BitInt(N){});


void ManglingInstantiator() {
  // LIN: define{{.*}} void @_Z20ManglingInstantiatorv()
  // WIN: define dso_local void @"?ManglingInstantiator@@YAXXZ"()
  _BitInt(93) A;
  ManglingTestTemplateParam(A);
// LIN: call void @_Z25ManglingTestTemplateParamIDB93_EvRT_(i93*
// WIN64: call void @"??$ManglingTestTemplateParam@U?$_BitInt@$0FN@@__clang@@@@YAXAEAU?$_BitInt@$0FN@@__clang@@@Z"(i93*
// WIN32: call void @"??$ManglingTestTemplateParam@U?$_BitInt@$0FN@@__clang@@@@YAXAAU?$_BitInt@$0FN@@__clang@@@Z"(i93*
  constexpr _BitInt(93) B = 993;
  ManglingTestNTTP<38>();
  // LIN: call void @_Z16ManglingTestNTTPILDB99_38EEvv()
  // WIN: call void @"??$ManglingTestNTTP@$0CG@@@YAXXZ"()
  ManglingTestNTTP<B>();
  // LIN: call void @_Z16ManglingTestNTTPILDB99_993EEvv()
  // WIN: call void @"??$ManglingTestNTTP@$0DOB@@@YAXXZ"()
  ManglingDependent<4>();
  // LIN: call signext i4 @_Z17ManglingDependentILi4EEDTtlDBT__EEv()
  // WIN64: call i4 @"??$ManglingDependent@$03@@YAU?$_BitInt@$03@__clang@@XZ"()
  // WIN32: call signext i4 @"??$ManglingDependent@$03@@YAU?$_BitInt@$03@__clang@@XZ"()
}

void TakesVarargs(int i, ...) {
  // LIN: define{{.*}} void @_Z12TakesVarargsiz(i32 %i, ...)
  // WIN: define dso_local void @"?TakesVarargs@@YAXHZZ"(i32 %i, ...)

  __builtin_va_list args;
  // LIN64: %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag]
  // LIN32: %[[ARGS:.+]] = alloca i8*
  // WIN: %[[ARGS:.+]] = alloca i8*
  __builtin_va_start(args, i);
  // LIN64: %[[STARTAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN64: %[[STARTAD1:.+]] = bitcast %struct.__va_list_tag* %[[STARTAD]] to i8*
  // LIN64: call void @llvm.va_start(i8* %[[STARTAD1]])
  // LIN32: %[[ARGSLLIFETIMESTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // LIN32: %[[ARGSSTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // LIN32: call void @llvm.va_start(i8* %[[ARGSSTART]])
  // WIN: %[[ARGSLLIFETIMESTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: %[[ARGSSTART:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: call void @llvm.va_start(i8* %[[ARGSSTART]])

  _BitInt(92) A = __builtin_va_arg(args, _BitInt(92));
  // LIN64: %[[AD1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN64: %[[OFA_P1:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD1]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P1]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 32
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi i92*
  // LIN64: %[[LOAD1:.+]] = load i92, i92* %[[BC1]]
  // LIN64: store i92 %[[LOAD1]], i92*

  // LIN32: %[[CUR1:.+]] = load i8*, i8** %[[ARGS]]
  // LIN32: %[[NEXT1:.+]] = getelementptr inbounds i8, i8* %[[CUR1]], i32 12
  // LIN32: store i8* %[[NEXT1]], i8** %[[ARGS]]
  // LIN32: %[[BC1:.+]] = bitcast i8* %[[CUR1]] to i92*
  // LIN32: %[[LOADV1:.+]] = load i92, i92* %[[BC1]]
  // LIN32: store i92 %[[LOADV1]], i92*

  // WIN64: %[[CUR1:.+]] = load i8*, i8** %[[ARGS]]
  // WIN64: %[[NEXT1:.+]] = getelementptr inbounds i8, i8* %[[CUR1]], i64 8
  // WIN64: store i8* %[[NEXT1]], i8** %[[ARGS]]
  // WIN64: %[[BC1:.+]] = bitcast i8* %[[CUR1]] to i92**
  // WIN64: %[[LOADP1:.+]] = load i92*, i92** %[[BC1]]
  // WIN64: %[[LOADV1:.+]] = load i92, i92* %[[LOADP1]]
  // WIN64: store i92 %[[LOADV1]], i92*

  // WIN32: %[[CUR1:.+]] = load i8*, i8** %[[ARGS]]
  // WIN32: %[[NEXT1:.+]] = getelementptr inbounds i8, i8* %[[CUR1]], i32 16 
  // WIN32: store i8* %[[NEXT1]], i8** %[[ARGS]]
  // WIN32: %[[BC1:.+]] = bitcast i8* %[[CUR1]] to i92*
  // WIN32: %[[LOADV1:.+]] = load i92, i92* %[[BC1]]
  // WIN32: store i92 %[[LOADV1]], i92*


  _BitInt(31) B = __builtin_va_arg(args, _BitInt(31));
  // LIN64: %[[AD2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN64: %[[OFA_P2:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD2]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P2]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi i31*
  // LIN64: %[[LOAD1:.+]] = load i31, i31* %[[BC1]]
  // LIN64: store i31 %[[LOAD1]], i31*

  // LIN32: %[[CUR2:.+]] = load i8*, i8** %[[ARGS]]
  // LIN32: %[[NEXT2:.+]] = getelementptr inbounds i8, i8* %[[CUR2]], i32 4
  // LIN32: store i8* %[[NEXT2]], i8** %[[ARGS]]
  // LIN32: %[[BC2:.+]] = bitcast i8* %[[CUR2]] to i31*
  // LIN32: %[[LOADV2:.+]] = load i31, i31* %[[BC2]]
  // LIN32: store i31 %[[LOADV2]], i31*

  // WIN64: %[[CUR2:.+]] = load i8*, i8** %[[ARGS]]
  // WIN64: %[[NEXT2:.+]] = getelementptr inbounds i8, i8* %[[CUR2]], i64 8
  // WIN64: store i8* %[[NEXT2]], i8** %[[ARGS]]
  // WIN64: %[[BC2:.+]] = bitcast i8* %[[CUR2]] to i31*
  // WIN64: %[[LOADV2:.+]] = load i31, i31* %[[BC2]]
  // WIN64: store i31 %[[LOADV2]], i31*

  // WIN32: %[[CUR2:.+]] = load i8*, i8** %[[ARGS]]
  // WIN32: %[[NEXT2:.+]] = getelementptr inbounds i8, i8* %[[CUR2]], i32 4
  // WIN32: store i8* %[[NEXT2]], i8** %[[ARGS]]
  // WIN32: %[[BC2:.+]] = bitcast i8* %[[CUR2]] to i31*
  // WIN32: %[[LOADV2:.+]] = load i31, i31* %[[BC2]]
  // WIN32: store i31 %[[LOADV2]], i31*

  _BitInt(16) C = __builtin_va_arg(args, _BitInt(16));
  // LIN64: %[[AD3:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN64: %[[OFA_P3:.+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %[[AD3]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, i32* %[[OFA_P3]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi i16*
  // LIN64: %[[LOAD1:.+]] = load i16, i16* %[[BC1]]
  // LIN64: store i16 %[[LOAD1]], i16*

  // LIN32: %[[CUR3:.+]] = load i8*, i8** %[[ARGS]]
  // LIN32: %[[NEXT3:.+]] = getelementptr inbounds i8, i8* %[[CUR3]], i32 4
  // LIN32: store i8* %[[NEXT3]], i8** %[[ARGS]]
  // LIN32: %[[BC3:.+]] = bitcast i8* %[[CUR3]] to i16*
  // LIN32: %[[LOADV3:.+]] = load i16, i16* %[[BC3]]
  // LIN32: store i16 %[[LOADV3]], i16*

  // WIN64: %[[CUR3:.+]] = load i8*, i8** %[[ARGS]]
  // WIN64: %[[NEXT3:.+]] = getelementptr inbounds i8, i8* %[[CUR3]], i64 8
  // WIN64: store i8* %[[NEXT3]], i8** %[[ARGS]]
  // WIN64: %[[BC3:.+]] = bitcast i8* %[[CUR3]] to i16*
  // WIN64: %[[LOADV3:.+]] = load i16, i16* %[[BC3]]
  // WIN64: store i16 %[[LOADV3]], i16*

  // WIN32: %[[CUR3:.+]] = load i8*, i8** %[[ARGS]]
  // WIN32: %[[NEXT3:.+]] = getelementptr inbounds i8, i8* %[[CUR3]], i32 4
  // WIN32: store i8* %[[NEXT3]], i8** %[[ARGS]]
  // WIN32: %[[BC3:.+]] = bitcast i8* %[[CUR3]] to i16*
  // WIN32: %[[LOADV3:.+]] = load i16, i16* %[[BC3]]
  // WIN32: store i16 %[[LOADV3]], i16*

  __builtin_va_end(args);
  // LIN64: %[[ENDAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %[[ARGS]]
  // LIN64: %[[ENDAD1:.+]] = bitcast %struct.__va_list_tag* %[[ENDAD]] to i8*
  // LIN64: call void @llvm.va_end(i8* %[[ENDAD1]])
  // LIN32: %[[ARGSEND:.+]] = bitcast i8** %[[ARGS]] to i8*
  // LIN32: call void @llvm.va_end(i8* %[[ARGSEND]])
  // WIN: %[[ARGSEND:.+]] = bitcast i8** %[[ARGS]] to i8*
  // WIN: call void @llvm.va_end(i8* %[[ARGSEND]])
}
void typeid_tests() {
  // LIN: define{{.*}} void @_Z12typeid_testsv()
  // WIN: define dso_local void @"?typeid_tests@@YAXXZ"()
  unsigned _BitInt(33) U33_1, U33_2;
  _BitInt(33) S33_1, S33_2;
  _BitInt(32) S32_1, S32_2;

 auto A = typeid(U33_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDU33_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDU33_ to %"class.std::type_info"*))
 // WIN64: call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UBitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32: call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UBitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto B = typeid(U33_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDU33_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDU33_ to %"class.std::type_info"*))
 // WIN64:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UBitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32:  call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor28* @"??_R0U?$_UBitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto C = typeid(S33_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDB33_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDB33_ to %"class.std::type_info"*))
 // WIN64:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32:  call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto D = typeid(S33_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDB33_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDB33_ to %"class.std::type_info"*))
 // WIN64:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32:  call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CB@@__clang@@@8" to %"class.std::type_info"*))
 auto E = typeid(S32_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDB32_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDB32_ to %"class.std::type_info"*))
 // WIN64:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32:  call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
 auto F = typeid(S32_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast ({ i8*, i8* }* @_ZTIDB32_ to %"class.std::type_info"*))
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast ({ i8*, i8* }* @_ZTIDB32_ to %"class.std::type_info"*))
 // WIN64:  call %"class.std::type_info"* @"??0type_info@std@@QEAA@AEBV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 8 dereferenceable(16) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
 // WIN32:  call x86_thiscallcc %"class.std::type_info"* @"??0type_info@std@@QAE@ABV01@@Z"(%"class.std::type_info"* {{[^,]*}} %{{.+}}, %"class.std::type_info"* nonnull align 4 dereferenceable(8) bitcast (%rtti.TypeDescriptor27* @"??_R0U?$_BitInt@$0CA@@__clang@@@8" to %"class.std::type_info"*))
}

void ExplicitCasts() {
  // LIN: define{{.*}} void @_Z13ExplicitCastsv()
  // WIN: define dso_local void @"?ExplicitCasts@@YAXXZ"()

  _BitInt(33) a;
  _BitInt(31) b;
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
  _BitInt(17) A;
  _BitInt(128) B;
  _BitInt(17) C;
};

void OffsetOfTest() {
  // LIN: define{{.*}} void @_Z12OffsetOfTestv()
  // WIN: define dso_local void @"?OffsetOfTest@@YAXXZ"()

  auto A = __builtin_offsetof(S,A);
  // CHECK: store i{{.+}} 0, i{{.+}}* %{{.+}}
  auto B = __builtin_offsetof(S,B);
  // LIN64: store i{{.+}} 8, i{{.+}}* %{{.+}}
  // LIN32: store i{{.+}} 4, i{{.+}}* %{{.+}}
  // WIN: store i{{.+}} 8, i{{.+}}* %{{.+}}
  auto C = __builtin_offsetof(S,C);
  // LIN64: store i{{.+}} 24, i{{.+}}* %{{.+}}
  // LIN32: store i{{.+}} 20, i{{.+}}* %{{.+}}
  // WIN: store i{{.+}} 24, i{{.+}}* %{{.+}}
}


void ShiftBitIntByConstant(_BitInt(28) Ext) {
// LIN: define{{.*}} void @_Z21ShiftBitIntByConstantDB28_
// WIN: define dso_local void @"?ShiftBitIntByConstant@@YAXU?$_BitInt@$0BM@@__clang@@@Z"
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

void ConstantShiftByBitInt(_BitInt(28) Ext, _BitInt(65) LargeExt) {
  // LIN: define{{.*}} void @_Z21ConstantShiftByBitIntDB28_DB65_
  // WIN: define dso_local void @"?ConstantShiftByBitInt@@YAXU?$_BitInt@$0BM@@__clang@@U?$_BitInt@$0EB@@2@@Z"
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

void Shift(_BitInt(28) Ext, _BitInt(65) LargeExt, int i) {
  // LIN: define{{.*}} void @_Z5ShiftDB28_DB65_
  // WIN: define dso_local void @"?Shift@@YAXU?$_BitInt@$0BM@@__clang@@U?$_BitInt@$0EB@@2@H@Z"
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

void ComplexTest(_Complex _BitInt(12) first, _Complex _BitInt(33) second) {
  // LIN: define{{.*}} void @_Z11ComplexTestCDB12_CDB33_
  // WIN: define dso_local void  @"?ComplexTest@@YAXU?$_Complex@U?$_BitInt@$0M@@__clang@@@__clang@@U?$_Complex@U?$_BitInt@$0CB@@__clang@@@2@@Z"
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
void TBAATest(_BitInt(sizeof(int) * 8) ExtInt,
              unsigned _BitInt(sizeof(int) * 8) ExtUInt,
              _BitInt(6) Other) {
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
// NoNewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{!"_BitInt(32)", ![[CHAR_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0}
// NoNewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{!"_BitInt(6)", ![[CHAR_TBAA_ROOT]], i64 0}

// NewStructPathTBAA-DAG: ![[CHAR_TBAA_ROOT:.+]] = !{![[TBAA_ROOT:.+]], i64 1, !"omnipotent char"}
// NewStructPathTBAA-DAG: ![[TBAA_ROOT]] = !{!"Simple C++ TBAA"}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA]] = !{![[EXTINT_TBAA_ROOT:.+]], ![[EXTINT_TBAA_ROOT]], i64 0, i64 4}
// NewStructPathTBAA-DAG: ![[EXTINT_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 4, !"_BitInt(32)"}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA]] = !{![[EXTINT6_TBAA_ROOT:.+]], ![[EXTINT6_TBAA_ROOT]], i64 0, i64 1}
// NewStructPathTBAA-DAG: ![[EXTINT6_TBAA_ROOT]] = !{![[CHAR_TBAA_ROOT]], i64 1, !"_BitInt(6)"}
