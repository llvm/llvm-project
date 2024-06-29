// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN64,NoNewStructPathTBAA
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN64,NewStructPathTBAA

// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN64,NoNewStructPathTBAA
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN64,NewStructPathTBAA

// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN32,NoNewStructPathTBAA
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,LIN,LIN32,NewStructPathTBAA

// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN32,NoNewStructPathTBAA
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -I%S -new-struct-path-tbaa -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,WIN,WIN32,NewStructPathTBAA

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
  // CHECK: %[[LOADA:.+]] = load i8, ptr %[[BFType:.*]]
  // CHECK: %[[CLEARA:.+]] = and i8 %[[LOADA]], -8
  // CHECK: %[[SETA:.+]] = or i8 %[[CLEARA]], 3
  // CHECK: %[[LOADB:.+]] = load i8, ptr %[[BFType:.*]]
  // CHECK: %[[CLEARB:.+]] = and i8 %[[LOADB]], -57
  // CHECK: %[[SETB:.+]] = or i8 %[[CLEARB]], 16
  // CHECK: %[[LOADC:.+]] = load i8, ptr %[[BFType:.*]]
  // CHECK: %[[CLEARC:.+]] = and i8 %[[LOADC]], 63
  // CHECK: %[[SETC:.+]] = or i8 %[[CLEARC]], 64
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

typedef unsigned _BitInt(16) uint16_t4 __attribute__((ext_vector_type(4)));
typedef _BitInt(32) vint32_t8 __attribute__((vector_size(32)));

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
// LIN: call void @_Z25ManglingTestTemplateParamIDB93_EvRT_(ptr
// WIN64: call void @"??$ManglingTestTemplateParam@U?$_BitInt@$0FN@@__clang@@@@YAXAEAU?$_BitInt@$0FN@@__clang@@@Z"(ptr
// WIN32: call void @"??$ManglingTestTemplateParam@U?$_BitInt@$0FN@@__clang@@@@YAXAAU?$_BitInt@$0FN@@__clang@@@Z"(ptr
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
  uint16_t4 V;
  ManglingTestTemplateParam(V);
  // LIN: call void @_Z25ManglingTestTemplateParamIDv4_DU16_EvRT_(ptr
  // WIN64: call void @"??$ManglingTestTemplateParam@T?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@@YAXAEAT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@Z"(ptr
  // WIN32: call void @"??$ManglingTestTemplateParam@T?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@@YAXAAT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@Z"(ptr

}

void TakesVarargs(int i, ...) {
  // LIN: define{{.*}} void @_Z12TakesVarargsiz(i32 %i, ...)
  // WIN: define dso_local void @"?TakesVarargs@@YAXHZZ"(i32 %i, ...)

  __builtin_va_list args;
  // LIN64: %[[ARGS:.+]] = alloca [1 x %struct.__va_list_tag]
  // LIN32: %[[ARGS:.+]] = alloca ptr
  // WIN: %[[ARGS:.+]] = alloca ptr
  __builtin_va_start(args, i);
  // LIN64: %[[STARTAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: call void @llvm.va_start.p0(ptr %[[STARTAD]])
  // LIN32: call void @llvm.va_start.p0(ptr %[[ARGS]])
  // WIN: call void @llvm.va_start.p0(ptr %[[ARGS]])

  _BitInt(92) A = __builtin_va_arg(args, _BitInt(92));
  // LIN64: %[[AD1:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: %[[OFA_P1:.+]] = getelementptr inbounds %struct.__va_list_tag, ptr %[[AD1]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, ptr %[[OFA_P1]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 32
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi ptr
  // LIN64: %[[LOAD1:.+]] = load i92, ptr %[[BC1]]
  // LIN64: store i92 %[[LOAD1]], ptr

  // LIN32: %[[CUR1:.+]] = load ptr, ptr %[[ARGS]]
  // LIN32: %[[NEXT1:.+]] = getelementptr inbounds i8, ptr %[[CUR1]], i32 12
  // LIN32: store ptr %[[NEXT1]], ptr %[[ARGS]]
  // LIN32: %[[LOADV1:.+]] = load i92, ptr %[[CUR1]]
  // LIN32: store i92 %[[LOADV1]], ptr

  // WIN64: %[[CUR1:.+]] = load ptr, ptr %[[ARGS]]
  // WIN64: %[[NEXT1:.+]] = getelementptr inbounds i8, ptr %[[CUR1]], i64 8
  // WIN64: store ptr %[[NEXT1]], ptr %[[ARGS]]
  // WIN64: %[[LOADP1:.+]] = load ptr, ptr %[[CUR1]]
  // WIN64: %[[LOADV1:.+]] = load i92, ptr %[[LOADP1]]
  // WIN64: store i92 %[[LOADV1]], ptr

  // WIN32: %[[CUR1:.+]] = load ptr, ptr %[[ARGS]]
  // WIN32: %[[NEXT1:.+]] = getelementptr inbounds i8, ptr %[[CUR1]], i32 16
  // WIN32: store ptr %[[NEXT1]], ptr %[[ARGS]]
  // WIN32: %[[LOADV1:.+]] = load i92, ptr %[[CUR1]]
  // WIN32: store i92 %[[LOADV1]], ptr


  _BitInt(31) B = __builtin_va_arg(args, _BitInt(31));
  // LIN64: %[[AD2:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: %[[OFA_P2:.+]] = getelementptr inbounds %struct.__va_list_tag, ptr %[[AD2]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, ptr %[[OFA_P2]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi ptr
  // LIN64: %[[LOAD1:.+]] = load i31, ptr %[[BC1]]
  // LIN64: store i31 %[[LOAD1]], ptr

  // LIN32: %[[CUR2:.+]] = load ptr, ptr %[[ARGS]]
  // LIN32: %[[NEXT2:.+]] = getelementptr inbounds i8, ptr %[[CUR2]], i32 4
  // LIN32: store ptr %[[NEXT2]], ptr %[[ARGS]]
  // LIN32: %[[LOADV2:.+]] = load i31, ptr %[[CUR2]]
  // LIN32: store i31 %[[LOADV2]], ptr

  // WIN64: %[[CUR2:.+]] = load ptr, ptr %[[ARGS]]
  // WIN64: %[[NEXT2:.+]] = getelementptr inbounds i8, ptr %[[CUR2]], i64 8
  // WIN64: store ptr %[[NEXT2]], ptr %[[ARGS]]
  // WIN64: %[[LOADV2:.+]] = load i31, ptr %[[CUR2]]
  // WIN64: store i31 %[[LOADV2]], ptr

  // WIN32: %[[CUR2:.+]] = load ptr, ptr %[[ARGS]]
  // WIN32: %[[NEXT2:.+]] = getelementptr inbounds i8, ptr %[[CUR2]], i32 4
  // WIN32: store ptr %[[NEXT2]], ptr %[[ARGS]]
  // WIN32: %[[LOADV2:.+]] = load i31, ptr %[[CUR2]]
  // WIN32: store i31 %[[LOADV2]], ptr

  _BitInt(16) C = __builtin_va_arg(args, _BitInt(16));
  // LIN64: %[[AD3:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: %[[OFA_P3:.+]] = getelementptr inbounds %struct.__va_list_tag, ptr %[[AD3]], i32 0, i32 0
  // LIN64: %[[GPOFFSET:.+]] = load i32, ptr %[[OFA_P3]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 40
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC1:.+]] = phi ptr
  // LIN64: %[[LOAD1:.+]] = load i16, ptr %[[BC1]]
  // LIN64: store i16 %[[LOAD1]], ptr

  // LIN32: %[[CUR3:.+]] = load ptr, ptr %[[ARGS]]
  // LIN32: %[[NEXT3:.+]] = getelementptr inbounds i8, ptr %[[CUR3]], i32 4
  // LIN32: store ptr %[[NEXT3]], ptr %[[ARGS]]
  // LIN32: %[[LOADV3:.+]] = load i16, ptr %[[CUR3]]
  // LIN32: store i16 %[[LOADV3]], ptr

  // WIN64: %[[CUR3:.+]] = load ptr, ptr %[[ARGS]]
  // WIN64: %[[NEXT3:.+]] = getelementptr inbounds i8, ptr %[[CUR3]], i64 8
  // WIN64: store ptr %[[NEXT3]], ptr %[[ARGS]]
  // WIN64: %[[LOADV3:.+]] = load i16, ptr %[[CUR3]]
  // WIN64: store i16 %[[LOADV3]], ptr

  // WIN32: %[[CUR3:.+]] = load ptr, ptr %[[ARGS]]
  // WIN32: %[[NEXT3:.+]] = getelementptr inbounds i8, ptr %[[CUR3]], i32 4
  // WIN32: store ptr %[[NEXT3]], ptr %[[ARGS]]
  // WIN32: %[[LOADV3:.+]] = load i16, ptr %[[CUR3]]
  // WIN32: store i16 %[[LOADV3]], ptr

  uint16_t4 D = __builtin_va_arg(args, uint16_t4);
  // LIN64: %[[AD4:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: %[[OFA_P4:.+]] = getelementptr inbounds %struct.__va_list_tag, ptr %[[AD4]], i32 0, i32 1
  // LIN64: %[[GPOFFSET:.+]] = load i32, ptr %[[OFA_P4]]
  // LIN64: %[[FITSINGP:.+]] = icmp ule i32 %[[GPOFFSET]], 160
  // LIN64: br i1 %[[FITSINGP]]
  // LIN64: %[[BC4:.+]] = phi ptr
  // LIN64: %[[LOADV4:.+]] = load <4 x i16>, ptr %[[BC4]]
  // LIN64: store <4 x i16> %[[LOADV4]], ptr

  // LIN32: %[[CUR4:.+]] = load ptr, ptr %[[ARGS]]
  // LIN32: %[[NEXT4:.+]] = getelementptr inbounds i8, ptr %[[CUR4]], i32 8
  // LIN32: store ptr %[[NEXT4]], ptr %[[ARGS]]
  // LIN32: %[[LOADV4:.+]] = load <4 x i16>, ptr %[[CUR4]]
  // LIN32: store <4 x i16> %[[LOADV4]], ptr %

  // WIN: %[[CUR4:.+]] = load ptr, ptr %[[ARGS]]
  // WIN64: %[[NEXT4:.+]] = getelementptr inbounds i8, ptr %[[CUR4]], i64 8
  // WIN32: %[[NEXT4:.+]] = getelementptr inbounds i8, ptr %[[CUR4]], i32 8
  // WIN: store ptr %[[NEXT4]], ptr %[[ARGS]]
  // WIN: %[[LOADV4:.+]] = load <4 x i16>, ptr %[[CUR4]]
  // WIN: store <4 x i16> %[[LOADV4]], ptr

  vint32_t8 E = __builtin_va_arg(args, vint32_t8);
  // LIN64: %[[AD5:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: %[[OFAA_P4:.+]] = getelementptr inbounds %struct.__va_list_tag, ptr %[[AD5]], i32 0, i32 2
  // LIN64: %[[OFAA:.+]] = load ptr, ptr %[[OFAA_P4]]

  // LIN64: [[OFAA_GEP:%.*]] = getelementptr inbounds i8, ptr %[[OFAA]], i32 31
  // LIN64: %[[OFAA_ALIGNED:.*]] = call ptr @llvm.ptrmask.p0.i64(ptr [[OFAA_GEP]], i64 -32)
  // LIN64: %[[LOADV5:.+]] = load <8 x i32>, ptr %[[OFAA_ALIGNED]]
  // LIN64: store <8 x i32> %[[LOADV5]], ptr

  // LIN32: %[[CUR5:.+]] = load ptr, ptr %[[ARGS]]

  // LIN32: [[GEP_CUR5:%.*]] = getelementptr inbounds i8, ptr %[[CUR5]], i32 31
  // LIN32: %[[CUR5_ALIGNED:.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[GEP_CUR5]], i32 -32)
  // LIN32: %[[NEXT5:.+]] = getelementptr inbounds i8, ptr %[[CUR5_ALIGNED]], i32 32
  // LIN32: store ptr %[[NEXT5]], ptr %[[ARGS]]
  // LIN32: %[[LOADV5:.+]] = load <8 x i32>, ptr %[[CUR5_ALIGNED]]
  // LIN32: store <8 x i32> %[[LOADV5]], ptr

  // WIN: %[[CUR5:.+]] = load ptr, ptr %[[ARGS]]
  // WIN64: %[[NEXT5:.+]] = getelementptr inbounds i8, ptr %[[CUR5]], i64 8
  // WIN32: %[[NEXT5:.+]] = getelementptr inbounds i8, ptr %[[CUR5]], i32 32
  // WIN: store ptr %[[NEXT5]], ptr %[[ARGS]]
  // WIN64: %[[LOADP5:.+]] = load ptr, ptr %[[CUR5]]
  // WIN64: %[[LOADV5:.+]] = load <8 x i32>, ptr %[[LOADP5]]
  // WIN32: %[[LOADV5:.+]] = load <8 x i32>, ptr %argp.cur7
  // WIN: store <8 x i32> %[[LOADV5]], ptr

  __builtin_va_end(args);
  // LIN64: %[[ENDAD:.+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[ARGS]]
  // LIN64: call void @llvm.va_end.p0(ptr %[[ENDAD]])
  // LIN32: call void @llvm.va_end.p0(ptr %[[ARGS]])
  // WIN: call void @llvm.va_end.p0(ptr %[[ARGS]])
}
void typeid_tests() {
  // LIN: define{{.*}} void @_Z12typeid_testsv()
  // WIN: define dso_local void @"?typeid_tests@@YAXXZ"()
  unsigned _BitInt(33) U33_1, U33_2;
  _BitInt(33) S33_1, S33_2;
  _BitInt(32) S32_1, S32_2;

 auto A = typeid(U33_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDU33_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDU33_)
 // WIN64: call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_UBitInt@$0CB@@__clang@@@8")
 // WIN32: call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_UBitInt@$0CB@@__clang@@@8")
 auto B = typeid(U33_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDU33_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDU33_)
 // WIN64:  call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_UBitInt@$0CB@@__clang@@@8")
 // WIN32:  call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_UBitInt@$0CB@@__clang@@@8")
 auto C = typeid(S33_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDB33_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDB33_)
 // WIN64:  call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_BitInt@$0CB@@__clang@@@8")
 // WIN32:  call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_BitInt@$0CB@@__clang@@@8")
 auto D = typeid(S33_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDB33_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDB33_)
 // WIN64:  call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_BitInt@$0CB@@__clang@@@8")
 // WIN32:  call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_BitInt@$0CB@@__clang@@@8")
 auto E = typeid(S32_1);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDB32_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDB32_)
 // WIN64:  call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_BitInt@$0CA@@__clang@@@8")
 // WIN32:  call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_BitInt@$0CA@@__clang@@@8")
 auto F = typeid(S32_2);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDB32_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDB32_)
 // WIN64:  call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0U?$_BitInt@$0CA@@__clang@@@8")
 // WIN32:  call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0U?$_BitInt@$0CA@@__clang@@@8")
 auto G = typeid(uint16_t4);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDv4_DU16_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDv4_DU16_)
 // WIN64: call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0T?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@8")
 // WIN32: call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0T?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@8")
 auto H = typeid(vint32_t8);
 // LIN64: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @_ZTIDv8_DB32_)
 // LIN32: call void @_ZNSt9type_infoC1ERKS_(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @_ZTIDv8_DB32_)
 // WIN64: call ptr @"??0type_info@std@@QEAA@AEBV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 8 dereferenceable(16) @"??_R0?AT?$__vector@U?$_BitInt@$0CA@@__clang@@$07@__clang@@@8")
 // WIN32: call x86_thiscallcc ptr @"??0type_info@std@@QAE@ABV01@@Z"(ptr {{[^,]*}} %{{.+}}, ptr nonnull align 4 dereferenceable(8) @"??_R0?AT?$__vector@U?$_BitInt@$0CA@@__clang@@$07@__clang@@@8")
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
  uint16_t4 c;
  c = i;
  // CHECK: %[[CONV:.+]] = trunc i32 %{{.+}} to i16
  // CHECK: %[[VEC:.+]] = insertelement <4 x i16> poison, i16 %[[CONV]], i64 0
  // CHECK: %[[Splat:.+]] = shufflevector <4 x i16> %[[VEC]], <4 x i16> poison, <4 x i32> zeroinitializer
}

struct S {
  _BitInt(17) A;
  _BitInt(128) B;
  _BitInt(17) C;
  uint16_t4 D;
  vint32_t8 E;
};

void OffsetOfTest() {
  // LIN: define{{.*}} void @_Z12OffsetOfTestv()
  // WIN: define dso_local void @"?OffsetOfTest@@YAXXZ"()

  auto A = __builtin_offsetof(S,A);
  // CHECK: store i{{.+}} 0, ptr %{{.+}}
  auto B = __builtin_offsetof(S,B);
  // LIN64: store i{{.+}} 8, ptr %{{.+}}
  // LIN32: store i{{.+}} 4, ptr %{{.+}}
  // WIN: store i{{.+}} 8, ptr %{{.+}}
  auto C = __builtin_offsetof(S,C);
  // LIN64: store i{{.+}} 24, ptr %{{.+}}
  // LIN32: store i{{.+}} 20, ptr %{{.+}}
  // WIN: store i{{.+}} 24, ptr %{{.+}}
  auto D = __builtin_offsetof(S,D);
  // LIN64: store i64 32, ptr %{{.+}}
  // LIN32: store i32 24, ptr %{{.+}}
  // WIN: store i{{.+}} 32, ptr %{{.+}}
  auto E = __builtin_offsetof(S,E);
  // LIN64: store i64 64, ptr %{{.+}}
  // LIN32: store i32 32, ptr %{{.+}}
  // WIN: store i{{.+}} 64, ptr %{{.+}}
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
void ShiftBitIntByConstant(uint16_t4 Ext) {
// LIN64: define{{.*}} void @_Z21ShiftBitIntByConstantDv4_DU16_(double %
// LIN32: define dso_local void @_Z21ShiftBitIntByConstantDv4_DU16_(i64 %
// WIN: define dso_local void @"?ShiftBitIntByConstant@@YAXT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@@Z"(<4 x i16>
  Ext << 7;
  // CHECK: shl <4 x i16> %{{.+}}, <i16 7, i16 7, i16 7, i16 7>
  Ext >> 7;
  // CHECK: lshr <4 x i16> %{{.+}}, <i16 7, i16 7, i16 7, i16 7>
  Ext << -7;
  // CHECK: shl <4 x i16> %{{.+}}, <i16 -7, i16 -7, i16 -7, i16 -7>
  Ext >> -7;
  // CHECK: lshr <4 x i16> %{{.+}}, <i16 -7, i16 -7, i16 -7, i16 -7>

  // UB in C/C++, Defined in OpenCL.
  Ext << 29;
  // CHECK: shl <4 x i16> %{{.+}}, <i16 29, i16 29, i16 29, i16 29>
  Ext >> 29;
  // CHECK: lshr <4 x i16> %{{.+}}, <i16 29, i16 29, i16 29, i16 29>
}
void ShiftBitIntByConstant(vint32_t8 Ext) {
// LIN64: define{{.*}} void @_Z21ShiftBitIntByConstantDv8_DB32_(ptr byval(<8 x i32>) align 32 %
// LIN32: define dso_local void @_Z21ShiftBitIntByConstantDv8_DB32_(<8 x i32> %
// WIN: define dso_local void @"?ShiftBitIntByConstant@@YAXT?$__vector@U?$_BitInt@$0CA@@__clang@@$07@__clang@@@Z"(<8 x i32>
  Ext << 7;
  // CHECK: shl <8 x i32> %{{.+}}, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  Ext >> 7;
  // CHECK: ashr <8 x i32> %{{.+}}, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  Ext << -7;
  // CHECK: shl <8 x i32> %{{.+}}, <i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7>
  Ext >> -7;
  // CHECK: ashr <8 x i32> %{{.+}}, <i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7, i32 -7>

  // UB in C/C++, Defined in OpenCL.
  Ext << 29;
  // CHECK: shl <8 x i32> %{{.+}}, <i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29>
  Ext >> 29;
  // CHECK: ashr <8 x i32> %{{.+}}, <i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29, i32 29>
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
  // CHECK: %[[FIRST_REALP:.+]] = getelementptr inbounds { i12, i12 }, ptr %{{.+}}, i32 0, i32 0
  // CHECK: %[[FIRST_REAL:.+]] = load i12, ptr %[[FIRST_REALP]]
  // CHECK: %[[FIRST_IMAGP:.+]] = getelementptr inbounds { i12, i12 }, ptr %{{.+}}, i32 0, i32 1
  // CHECK: %[[FIRST_IMAG:.+]] = load i12, ptr %[[FIRST_IMAGP]]
  // CHECK: %[[FIRST_REAL_CONV:.+]] = sext i12 %[[FIRST_REAL]]
  // CHECK: %[[FIRST_IMAG_CONV:.+]] = sext i12 %[[FIRST_IMAG]]
  // CHECK: %[[SECOND_REALP:.+]] = getelementptr inbounds { i33, i33 }, ptr %{{.+}}, i32 0, i32 0
  // CHECK: %[[SECOND_REAL:.+]] = load i33, ptr %[[SECOND_REALP]]
  // CHECK: %[[SECOND_IMAGP:.+]] = getelementptr inbounds { i33, i33 }, ptr %{{.+}}, i32 0, i32 1
  // CHECK: %[[SECOND_IMAG:.+]] = load i33, ptr %[[SECOND_IMAGP]]
  // CHECK: %[[REAL:.+]] = add i33 %[[FIRST_REAL_CONV]], %[[SECOND_REAL]]
  // CHECK: %[[IMAG:.+]] = add i33 %[[FIRST_IMAG_CONV]], %[[SECOND_IMAG]]
}

typedef  _BitInt(64) vint64_t16 __attribute__((vector_size(16)));
void VectorTest(vint64_t16 first, vint64_t16 second) {
  // LIN: define{{.*}} void @_Z10VectorTestDv2_DB64_S0_(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // WIN64: define dso_local void @"?VectorTest@@YAXT?$__vector@U?$_BitInt@$0EA@@__clang@@$01@__clang@@0@Z"(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // WIN32: define dso_local void @"?VectorTest@@YAXT?$__vector@U?$_BitInt@$0EA@@__clang@@$01@__clang@@0@Z"(<2 x i64> inreg %{{.+}}, <2 x i64> inreg %{{.+}})
  __builtin_shufflevector (first, first, 1, 3, 2) + __builtin_shufflevector (second, second, 1, 3, 2);
  // CHECK: %[[Shuffle:.+]] = shufflevector <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <3 x i32> <i32 1, i32 3, i32 2>
  // CHECK:  %[[Shuffle1:.+]] = shufflevector <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <3 x i32> <i32 1, i32 3, i32 2>
  // CHECK: %[[ADD:.+]] = add <3 x i64> %[[Shuffle]], %[[Shuffle1]]
}

void VectorTest(uint16_t4 first, uint16_t4 second) {
  // LIN64: define{{.*}} void @_Z10VectorTestDv4_DU16_S0_(double %{{.+}}, double %{{.+}})
  // LIN32: define{{.*}} void @_Z10VectorTestDv4_DU16_S0_(i64 %{{.+}}, i64 %{{.+}})
  // WIN64: define dso_local void @"?VectorTest@@YAXT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@0@Z"(<4 x i16> %{{.+}}, <4 x i16> %{{.+}})
  // WIN32: define dso_local void @"?VectorTest@@YAXT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@0@Z"(<4 x i16> inreg %{{.+}}, <4 x i16> inreg %{{.+}})
  first.xzw + second.zwx;
  // CHECK: %[[Shuffle:.+]] = shufflevector <4 x i16> %{{.+}}, <4 x i16> poison, <3 x i32> <i32 0, i32 2, i32 3>
  // CHECK: %[[Shuffle1:.+]] = shufflevector <4 x i16> %{{.+}}, <4 x i16> poison, <3 x i32> <i32 2, i32 3, i32 0>
  // CHECK: %[[ADD:.+]] = add <3 x i16> %[[Shuffle]], %[[Shuffle1]]
}

// Ensure that these types don't alias the normal int types.
void TBAATest(_BitInt(sizeof(int) * 8) ExtInt,
              unsigned _BitInt(sizeof(int) * 8) ExtUInt,
              _BitInt(6) Other) {
  // CHECK-DAG: store i32 %{{.+}}, ptr %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA:.+]]
  // CHECK-DAG: store i32 %{{.+}}, ptr %{{.+}}, align 4, !tbaa ![[EXTINT_TBAA]]
  // CHECK-DAG: store i6 %{{.+}}, ptr %{{.+}}, align 1, !tbaa ![[EXTINT6_TBAA:.+]]
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
