// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=CXXEH

extern "C" int basic_filter(int v, ...);
extern "C" void might_crash();

extern "C" void test_freefunc(int p1) {
  int l1 = 13;
  static int s1 = 42;
  __try {
    might_crash();
  } __except(basic_filter(p1, l1, s1)) {
  }
}

// CHECK-LABEL: define dso_local void @test_freefunc(i32 %p1)
// CHECK: @llvm.localescape(i32* %[[p1_ptr:[^, ]*]], i32* %[[l1_ptr:[^, ]*]])
// CHECK: store i32 %p1, i32* %[[p1_ptr]], align 4
// CHECK: store i32 13, i32* %[[l1_ptr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal i32 @"?filt$0@0@test_freefunc@@"(i8* %exception_pointers, i8* %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %frame_pointer)
// CHECK: %[[p1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[p1_ptr:[^ ]*]] = bitcast i8* %[[p1_i8]] to i32*
// CHECK: %[[l1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %[[fp]], i32 1)
// CHECK: %[[l1_ptr:[^ ]*]] = bitcast i8* %[[l1_i8]] to i32*
// CHECK: %[[s1:[^ ]*]] = load i32, i32* @"?s1@?1??test_freefunc@@9@4HA", align 4
// CHECK: %[[l1:[^ ]*]] = load i32, i32* %[[l1_ptr]]
// CHECK: %[[p1:[^ ]*]] = load i32, i32* %[[p1_ptr]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 %[[p1]], i32 %[[l1]], i32 %[[s1]])

struct S {
  int m1;
  void test_method(void);
};

void S::test_method() {
  int l1 = 13;
  __try {
    might_crash();
  } __except(basic_filter(l1)) {
    // FIXME: Test capturing 'this' and 'm1'.
  }
}

// CHECK-LABEL: define dso_local void @"?test_method@S@@QEAAXXZ"(%struct.S* %this)
// CHECK: @llvm.localescape(i32* %[[l1_addr:[^, ]*]])
// CHECK: store i32 13, i32* %[[l1_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal i32 @"?filt$0@0@test_method@S@@"(i8* %exception_pointers, i8* %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (%struct.S*)* @"?test_method@S@@QEAAXXZ" to i8*), i8* %frame_pointer)
// CHECK: %[[l1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%struct.S*)* @"?test_method@S@@QEAAXXZ" to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[l1_ptr:[^ ]*]] = bitcast i8* %[[l1_i8]] to i32*
// CHECK: %[[l1:[^ ]*]] = load i32, i32* %[[l1_ptr]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 %[[l1]])

void test_lambda() {
  int l1 = 13;
  auto lambda = [&]() {
    int l2 = 42;
    __try {
      might_crash();
    } __except(basic_filter(l2)) {
      // FIXME: Test 'l1' when we can capture the lambda's 'this' decl.
    }
  };
  lambda();
}

// CHECK-LABEL: define internal void @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ"(%class.anon* %this)
// CHECK: @llvm.localescape(i32* %[[l2_addr:[^, ]*]])
// CHECK: store i32 42, i32* %[[l2_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal i32 @"?filt$0@0@?R<lambda_0>@?0??test_lambda@@YAXXZ@"(i8* %exception_pointers, i8* %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (%class.anon*)* @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ" to i8*), i8* %frame_pointer)
// CHECK: %[[l2_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%class.anon*)* @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ" to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[l2_ptr:[^ ]*]] = bitcast i8* %[[l2_i8]] to i32*
// CHECK: %[[l2:[^ ]*]] = load i32, i32* %[[l2_ptr]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 %[[l2]])
