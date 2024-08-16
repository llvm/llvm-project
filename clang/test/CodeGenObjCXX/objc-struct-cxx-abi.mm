// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fobjc-arc  -fobjc-weak -fobjc-runtime-has-weak -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fobjc-arc  -fobjc-weak -fobjc-runtime-has-weak -fclang-abi-compat=4.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fobjc-arc  -fobjc-weak -fobjc-runtime-has-weak -emit-llvm -o - -DTRIVIALABI %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fobjc-arc  -fobjc-weak -fobjc-runtime-has-weak -fclang-abi-compat=4.0 -emit-llvm -o - -DTRIVIALABI %s | FileCheck %s

// Check that structs consisting solely of __strong or __weak pointer fields are
// destructed in the callee function and structs consisting solely of __strong
// pointer fields are passed directly.

// CHECK: %[[STRUCT_STRONGWEAK:.*]] = type { ptr, ptr }
// CHECK: %[[STRUCT_STRONG:.*]] = type { ptr }
// CHECK: %[[STRUCT_CONTAINSNONTRIVIAL:.*]] = type { %{{.*}}, ptr }
// CHECK: %[[STRUCT_NONTRIVIAL:.*]] = type { ptr }
// CHECK: %[[STRUCT_CONTAINSSTRONGWEAK:.*]] = type { %[[STRUCT_STRONGWEAK]] }
// CHECK: %[[STRUCT_S:.*]] = type { ptr }

#ifdef TRIVIALABI
struct __attribute__((trivial_abi)) StrongWeak {
#else
struct StrongWeak {
#endif
  id fstrong;
  __weak id fweak;
};

#ifdef TRIVIALABI
struct __attribute__((trivial_abi)) ContainsStrongWeak {
#else
struct ContainsStrongWeak {
#endif
  StrongWeak sw;
};

#ifdef TRIVIALABI
struct __attribute__((trivial_abi)) DerivedStrongWeak : StrongWeak {
#else
struct DerivedStrongWeak : StrongWeak {
#endif
};

#ifdef TRIVIALABI
struct __attribute__((trivial_abi)) Strong {
#else
struct Strong {
#endif
  id fstrong;
};

template<class T>
#ifdef TRIVIALABI
struct __attribute__((trivial_abi)) S {
#else
struct S {
#endif
  T a;
};

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &);
  ~NonTrivial();
  int *a;
};

// This struct is not passed directly nor destructed in the callee because f0
// has type NonTrivial.
struct ContainsNonTrivial {
  NonTrivial f0;
  id f1;
};

@interface C
- (void)passStrong:(Strong)a;
- (void)passStrongWeak:(StrongWeak)a;
- (void)passNonTrivial:(NonTrivial)a;
@end

// CHECK: define{{.*}} void @_Z19testParamStrongWeak10StrongWeak(ptr noundef %{{.*}})
// CHECK: call noundef ptr @_ZN10StrongWeakD1Ev(
// CHECK-NEXT: ret void

void testParamStrongWeak(StrongWeak a) {
}

// CHECK: define{{.*}} void @_Z18testCallStrongWeakP10StrongWeak(ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONGWEAK]], align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[CALL:.*]] = call noundef ptr @_ZN10StrongWeakC1ERKS_(ptr {{[^,]*}} %[[AGG_TMP]], ptr noundef nonnull align 8 dereferenceable(16) %[[V0]])
// CHECK: call void @_Z19testParamStrongWeak10StrongWeak(ptr noundef %[[AGG_TMP]])
// CHECK-NOT: call
// CHECK: ret void

void testCallStrongWeak(StrongWeak *a) {
  testParamStrongWeak(*a);
}

// CHECK: define{{.*}} void @_Z20testReturnStrongWeakP10StrongWeak(ptr dead_on_unwind noalias writable sret(%[[STRUCT_STRONGWEAK:.*]]) align 8 %[[AGG_RESULT:.*]], ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:a.addr]] = alloca ptr, align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[CALL:.*]] = call noundef ptr @_ZN10StrongWeakC1ERKS_(ptr {{[^,]*}} %[[AGG_RESULT]], ptr noundef nonnull align 8 dereferenceable(16) %[[V0]])
// CHECK: ret void

StrongWeak testReturnStrongWeak(StrongWeak *a) {
  return *a;
}

// CHECK: define{{.*}} void @_Z27testParamContainsStrongWeak18ContainsStrongWeak(ptr noundef %[[A:.*]])
// CHECK: call noundef ptr @_ZN18ContainsStrongWeakD1Ev(ptr {{[^,]*}} %[[A]])

void testParamContainsStrongWeak(ContainsStrongWeak a) {
}

// CHECK: define{{.*}} void @_Z26testParamDerivedStrongWeak17DerivedStrongWeak(ptr noundef %[[A:.*]])
// CHECK: call noundef ptr @_ZN17DerivedStrongWeakD1Ev(ptr {{[^,]*}} %[[A]])

void testParamDerivedStrongWeak(DerivedStrongWeak a) {
}

// CHECK: define{{.*}} void @_Z15testParamStrong6Strong(i64 %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[A]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[A_COERCE]] to ptr
// CHECK: store ptr %[[COERCE_VAL_IP]], ptr %[[COERCE_DIVE]], align 8
// CHECK: %[[CALL:.*]] = call noundef ptr @_ZN6StrongD1Ev(ptr {{[^,]*}} %[[A]])
// CHECK: ret void

// CHECK: define linkonce_odr noundef ptr @_ZN6StrongD1Ev(

void testParamStrong(Strong a) {
}

// CHECK: define{{.*}} void @_Z14testCallStrongP6Strong(ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[CALL:.*]] = call noundef ptr @_ZN6StrongC1ERKS_(ptr {{[^,]*}} %[[AGG_TMP]], ptr noundef nonnull align 8 dereferenceable(8) %[[V0]])
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint ptr %[[V1]] to i64
// CHECK: call void @_Z15testParamStrong6Strong(i64 %[[COERCE_VAL_PI]])
// CHECK: ret void

void testCallStrong(Strong *a) {
  testParamStrong(*a);
}

// CHECK: define{{.*}} i64 @_Z16testReturnStrongP6Strong(ptr noundef %[[A:.*]])
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: %[[CALL:.*]] = call noundef ptr @_ZN6StrongC1ERKS_(ptr {{[^,]*}} %[[RETVAL]], ptr noundef nonnull align 8 dereferenceable(8) %[[V0]])
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint ptr %[[V1]] to i64
// CHECK: ret i64 %[[COERCE_VAL_PI]]

Strong testReturnStrong(Strong *a) {
  return *a;
}

// CHECK: define{{.*}} void @_Z21testParamWeakTemplate1SIU6__weakP11objc_objectE(ptr noundef %{{.*}})
// CHECK: call noundef ptr @_ZN1SIU6__weakP11objc_objectED1Ev(
// CHECK-NEXT: ret void

void testParamWeakTemplate(S<__weak id> a) {
}

// CHECK: define{{.*}} void @_Z27testParamContainsNonTrivial18ContainsNonTrivial(ptr noundef %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void testParamContainsNonTrivial(ContainsNonTrivial a) {
}

// CHECK: define{{.*}} void @_Z26testCallContainsNonTrivialP18ContainsNonTrivial(
// CHECK: call void @_Z27testParamContainsNonTrivial18ContainsNonTrivial(ptr noundef %{{.*}})
// CHECK: call noundef ptr @_ZN18ContainsNonTrivialD1Ev(ptr {{[^,]*}} %{{.*}})

void testCallContainsNonTrivial(ContainsNonTrivial *a) {
  testParamContainsNonTrivial(*a);
}

namespace testThunk {

// CHECK-LABEL: define{{.*}} i64 @_ZThn8_N9testThunk2D02m0Ev(
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[CALL:.*]] = tail call i64 @_ZN9testThunk2D02m0Ev(
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[RETVAL]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[CALL]] to ptr
// CHECK: store ptr %[[COERCE_VAL_IP]], ptr %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_DIVE2:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V3:.*]] = load ptr, ptr %[[COERCE_DIVE2]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint ptr %[[V3]] to i64
// CHECK: ret i64 %[[COERCE_VAL_PI]]

struct B0 {
  virtual Strong m0();
};

struct B1 {
  virtual Strong m0();
};

struct D0 : B0, B1 {
  Strong m0() override;
};

Strong D0::m0() { return {}; }

}

namespace testNullReceiver {

// CHECK-LABEL: define{{.*}} void @_ZN16testNullReceiver5test0EP1C(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: br i1

// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[STRUCT_STRONG]], ptr %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V7:.*]] = load ptr, ptr %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint ptr %[[V7]] to i64
// CHECK: call void @objc_msgSend({{.*}}, i64 %[[COERCE_VAL_PI]])
// CHECK: br

// CHECK: %[[CALL1:.*]] = call noundef ptr @_ZN6StrongD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %[[AGG_TMP]])
// CHECK: br

void test0(C *c) {
  [c passStrong:Strong()];
}

// CHECK-LABEL: define{{.*}} void @_ZN16testNullReceiver5test1EP1C(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_STRONGWEAK]], align 8
// CHECK: br i1

// CHECK: call void @objc_msgSend({{.*}}, ptr noundef %[[AGG_TMP]])
// CHECK: br

// CHECK: %[[CALL1:.*]] = call noundef ptr @_ZN10StrongWeakD1Ev(ptr noundef nonnull align 8 dereferenceable(16) %[[AGG_TMP]])
// CHECK: br

void test1(C *c) {
  [c passStrongWeak:StrongWeak()];
}

// No null check needed.

// CHECK-LABEL: define{{.*}} void @_ZN16testNullReceiver5test2EP1C(
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_NONTRIVIAL]], align 8
// CHECK: call void @objc_msgSend({{.*}}, ptr noundef %[[AGG_TMP]])
// CHECK-NEXT: call noundef ptr @_ZN10NonTrivialD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %[[AGG_TMP]])

void test2(C *c) {
  [c passNonTrivial:NonTrivial()];
}

}
