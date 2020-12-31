// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios11 -std=c++11 -fcxx-exceptions -fexceptions -fclang-abi-compat=4.0 -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_SMALL:.*]] = type { i32* }
// CHECK: %[[STRUCT_LARGE:.*]] = type { i32*, [128 x i32] }
// CHECK: %[[STRUCT_TRIVIAL:.*]] = type { i32 }
// CHECK: %[[STRUCT_NONTRIVIAL:.*]] = type { i32 }

struct __attribute__((trivial_abi)) Small {
  int *p;
  Small();
  ~Small();
  Small(const Small &) noexcept;
  Small &operator=(const Small &);
};

struct __attribute__((trivial_abi)) Large {
  int *p;
  int a[128];
  Large();
  ~Large();
  Large(const Large &) noexcept;
  Large &operator=(const Large &);
};

struct Trivial {
  int a;
};

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
  int a;
};

struct HasTrivial {
  Small s;
  Trivial m;
};

struct HasNonTrivial {
  Small s;
  NonTrivial m;
};

struct B0 {
  virtual Small m0();
};

struct B1 {
  virtual Small m0();
};

struct D0 : B0, B1 {
  Small m0() override;
};

// CHECK-LABEL: define{{.*}} i64 @_ZThn8_N2D02m0Ev(
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_SMALL]], align 8
// CHECK: %[[CALL:.*]] = tail call i64 @_ZN2D02m0Ev(
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[RETVAL]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[CALL]] to i32*
// CHECK: store i32* %[[COERCE_VAL_IP]], i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_DIVE2:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V3:.*]] = load i32*, i32** %[[COERCE_DIVE2]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint i32* %[[V3]] to i64
// CHECK: ret i64 %[[COERCE_VAL_PI]]

Small D0::m0() { return {}; }

// CHECK: define{{.*}} void @_Z14testParamSmall5Small(i64 %[[A_COERCE:.*]])
// CHECK: %[[A:.*]] = alloca %[[STRUCT_SMALL]], align 8
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[A]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[A_COERCE]] to i32*
// CHECK: store i32* %[[COERCE_VAL_IP]], i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[CALL:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallD1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[A]])
// CHECK: ret void
// CHECK: }

void testParamSmall(Small a) noexcept {
}

// CHECK: define{{.*}} i64 @_Z15testReturnSmallv()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_SMALL:.*]], align 8
// CHECK: %[[CALL:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallC1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[RETVAL]])
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V0:.*]] = load i32*, i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint i32* %[[V0]] to i64
// CHECK: ret i64 %[[COERCE_VAL_PI]]
// CHECK: }

Small testReturnSmall() {
  Small t;
  return t;
}

// CHECK: define{{.*}} void @_Z14testCallSmall0v()
// CHECK: %[[T:.*]] = alloca %[[STRUCT_SMALL:.*]], align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SMALL]], align 8
// CHECK: %[[CALL:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallC1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[T]])
// CHECK: %[[CALL1:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallC1ERKS_(%[[STRUCT_SMALL]]* {{[^,]*}} %[[AGG_TMP]], %[[STRUCT_SMALL]]* nonnull align 8 dereferenceable(8) %[[T]])
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V0:.*]] = load i32*, i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint i32* %[[V0]] to i64
// CHECK: call void @_Z14testParamSmall5Small(i64 %[[COERCE_VAL_PI]])
// CHECK: %[[CALL2:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallD1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[T]])
// CHECK: ret void
// CHECK: }

void testCallSmall0() {
  Small t;
  testParamSmall(t);
}

// CHECK: define{{.*}} void @_Z14testCallSmall1v()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SMALL:.*]], align 8
// CHECK: %[[CALL:.*]] = call i64 @_Z15testReturnSmallv()
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[CALL]] to i32*
// CHECK: store i32* %[[COERCE_VAL_IP]], i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[COERCE_DIVE1:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[AGG_TMP]], i32 0, i32 0
// CHECK: %[[V0:.*]] = load i32*, i32** %[[COERCE_DIVE1]], align 8
// CHECK: %[[COERCE_VAL_PI:.*]] = ptrtoint i32* %[[V0]] to i64
// CHECK: call void @_Z14testParamSmall5Small(i64 %[[COERCE_VAL_PI]])
// CHECK: ret void
// CHECK: }

void testCallSmall1() {
  testParamSmall(testReturnSmall());
}

// CHECK: define{{.*}} void @_Z16testIgnoredSmallv()
// CHECK: %[[AGG_TMP_ENSURED:.*]] = alloca %[[STRUCT_SMALL:.*]], align 8
// CHECK: %[[CALL:.*]] = call i64 @_Z15testReturnSmallv()
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_SMALL]], %[[STRUCT_SMALL]]* %[[AGG_TMP_ENSURED]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[CALL]] to i32*
// CHECK: store i32* %[[COERCE_VAL_IP]], i32** %[[COERCE_DIVE]], align 8
// CHECK: %[[CALL1:.*]] = call %[[STRUCT_SMALL]]* @_ZN5SmallD1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[AGG_TMP_ENSURED]])
// CHECK: ret void
// CHECK: }

void testIgnoredSmall() {
  testReturnSmall();
}

// CHECK: define{{.*}} void @_Z14testParamLarge5Large(%[[STRUCT_LARGE:.*]]* %[[A:.*]])
// CHECK: %[[CALL:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeD1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[A]])
// CHECK: ret void
// CHECK: }

void testParamLarge(Large a) noexcept {
}

// CHECK: define{{.*}} void @_Z15testReturnLargev(%[[STRUCT_LARGE:.*]]* noalias sret(%[[STRUCT_LARGE]]) align 8 %[[AGG_RESULT:.*]])
// CHECK: %[[CALL:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeC1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_RESULT]])
// CHECK: ret void
// CHECK: }

Large testReturnLarge() {
  Large t;
  return t;
}

// CHECK: define{{.*}} void @_Z14testCallLarge0v()
// CHECK: %[[T:.*]] = alloca %[[STRUCT_LARGE:.*]], align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_LARGE]], align 8
// CHECK: %[[CALL:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeC1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[T]])
// CHECK: %[[CALL1:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeC1ERKS_(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_TMP]], %[[STRUCT_LARGE]]* nonnull align 8 dereferenceable(520) %[[T]])
// CHECK: call void @_Z14testParamLarge5Large(%[[STRUCT_LARGE]]* %[[AGG_TMP]])
// CHECK: %[[CALL2:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeD1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[T]])
// CHECK: ret void
// CHECK: }

void testCallLarge0() {
  Large t;
  testParamLarge(t);
}

// CHECK: define{{.*}} void @_Z14testCallLarge1v()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_LARGE:.*]], align 8
// CHECK: call void @_Z15testReturnLargev(%[[STRUCT_LARGE]]* sret(%[[STRUCT_LARGE]]) align 8 %[[AGG_TMP]])
// CHECK: call void @_Z14testParamLarge5Large(%[[STRUCT_LARGE]]* %[[AGG_TMP]])
// CHECK: ret void
// CHECK: }

void testCallLarge1() {
  testParamLarge(testReturnLarge());
}

// CHECK: define{{.*}} void @_Z16testIgnoredLargev()
// CHECK: %[[AGG_TMP_ENSURED:.*]] = alloca %[[STRUCT_LARGE:.*]], align 8
// CHECK: call void @_Z15testReturnLargev(%[[STRUCT_LARGE]]* sret(%[[STRUCT_LARGE]]) align 8 %[[AGG_TMP_ENSURED]])
// CHECK: %[[CALL:.*]] = call %[[STRUCT_LARGE]]* @_ZN5LargeD1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_TMP_ENSURED]])
// CHECK: ret void
// CHECK: }

void testIgnoredLarge() {
  testReturnLarge();
}

// CHECK: define{{.*}} i64 @_Z20testReturnHasTrivialv()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_TRIVIAL:.*]], align 4
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_TRIVIAL]], %[[STRUCT_TRIVIAL]]* %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V0:.*]] = load i32, i32* %[[COERCE_DIVE]], align 4
// CHECK: %[[COERCE_VAL_II:.*]] = zext i32 %[[V0]] to i64
// CHECK: ret i64 %[[COERCE_VAL_II]]
// CHECK: }

Trivial testReturnHasTrivial() {
  Trivial t;
  return t;
}

// CHECK: define{{.*}} void @_Z23testReturnHasNonTrivialv(%[[STRUCT_NONTRIVIAL:.*]]* noalias sret(%[[STRUCT_NONTRIVIAL]]) align 4 %[[AGG_RESULT:.*]])
// CHECK: %[[CALL:.*]] = call %[[STRUCT_NONTRIVIAL]]* @_ZN10NonTrivialC1Ev(%[[STRUCT_NONTRIVIAL]]* {{[^,]*}} %[[AGG_RESULT]])
// CHECK: ret void
// CHECK: }

NonTrivial testReturnHasNonTrivial() {
  NonTrivial t;
  return t;
}

// CHECK: define{{.*}} void @_Z18testExceptionSmallv()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SMALL]], align 8
// CHECK: %[[AGG_TMP1:.*]] = alloca %[[STRUCT_SMALL]], align 8
// CHECK: call %[[STRUCT_SMALL]]* @_ZN5SmallC1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[AGG_TMP]])
// CHECK: invoke %[[STRUCT_SMALL]]* @_ZN5SmallC1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[AGG_TMP1]])

// CHECK: call void @_Z20calleeExceptionSmall5SmallS_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT: ret void

// CHECK: landingpad { i8*, i32 }
// CHECK: call %[[STRUCT_SMALL]]* @_ZN5SmallD1Ev(%[[STRUCT_SMALL]]* {{[^,]*}} %[[AGG_TMP]])
// CHECK: br

// CHECK: resume { i8*, i32 }

void calleeExceptionSmall(Small, Small);

void testExceptionSmall() {
  calleeExceptionSmall(Small(), Small());
}

// CHECK: define{{.*}} void @_Z18testExceptionLargev()
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_LARGE]], align 8
// CHECK: %[[AGG_TMP1:.*]] = alloca %[[STRUCT_LARGE]], align 8
// CHECK: call %[[STRUCT_LARGE]]* @_ZN5LargeC1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_TMP]])
// CHECK: invoke %[[STRUCT_LARGE]]* @_ZN5LargeC1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_TMP1]])

// CHECK: call void @_Z20calleeExceptionLarge5LargeS_(%[[STRUCT_LARGE]]* %[[AGG_TMP]], %[[STRUCT_LARGE]]* %[[AGG_TMP1]])
// CHECK-NEXT: ret void

// CHECK: landingpad { i8*, i32 }
// CHECK: call %[[STRUCT_LARGE]]* @_ZN5LargeD1Ev(%[[STRUCT_LARGE]]* {{[^,]*}} %[[AGG_TMP]])
// CHECK: br

// CHECK: resume { i8*, i32 }

void calleeExceptionLarge(Large, Large);

void testExceptionLarge() {
  calleeExceptionLarge(Large(), Large());
}
