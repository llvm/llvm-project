// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | FileCheck %s

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

// CHECK-LABEL: define dso_local void @test_freefunc(i32 noundef %p1)
// CHECK: @llvm.localescape(i32* %[[p1_ptr:[^, ]*]], i32* %[[l1_ptr:[^, ]*]])
// CHECK: store i32 %p1, i32* %[[p1_ptr]], align 4
// CHECK: store i32 13, i32* %[[l1_ptr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@test_freefunc@@"(i8* noundef %exception_pointers, i8* noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %frame_pointer)
// CHECK: %[[p1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[p1_ptr:[^ ]*]] = bitcast i8* %[[p1_i8]] to i32*
// CHECK: %[[l1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (i32)* @test_freefunc to i8*), i8* %[[fp]], i32 1)
// CHECK: %[[l1_ptr:[^ ]*]] = bitcast i8* %[[l1_i8]] to i32*
// CHECK: %[[s1:[^ ]*]] = load i32, i32* @"?s1@?1??test_freefunc@@9@4HA", align 4
// CHECK: %[[l1:[^ ]*]] = load i32, i32* %[[l1_ptr]]
// CHECK: %[[p1:[^ ]*]] = load i32, i32* %[[p1_ptr]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 noundef %[[p1]], i32 noundef %[[l1]], i32 noundef %[[s1]])

struct S {
  int m1;
  void test_method(void);
};

void S::test_method() {
  int l1 = 13;
  __try {
    might_crash();
  } __except (basic_filter(l1, m1)) {
  }
}

// CHECK-LABEL: define dso_local void @"?test_method@S@@QEAAXXZ"(%struct.S* {{[^,]*}} %this)
// CHECK: @llvm.localescape(i32* %[[l1_addr:[^, ]*]], %struct.S** %[[this_addr:[^, ]*]])
// CHECK: store %struct.S* %this, %struct.S** %[[this_addr]], align 8
// CHECK: store i32 13, i32* %[[l1_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@test_method@S@@"(i8* noundef %exception_pointers, i8* noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (%struct.S*)* @"?test_method@S@@QEAAXXZ" to i8*), i8* %frame_pointer)
// CHECK: %[[l1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%struct.S*)* @"?test_method@S@@QEAAXXZ" to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[l1_ptr:[^ ]*]] = bitcast i8* %[[l1_i8]] to i32*
// CHECK: %[[this_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%struct.S*)* @"?test_method@S@@QEAAXXZ" to i8*), i8* %[[fp]], i32 1)
// CHECK: %[[this_ptr:[^ ]*]] = bitcast i8* %[[this_i8]] to %struct.S**
// CHECK: %[[this:[^ ]*]] = load %struct.S*, %struct.S** %[[this_ptr]], align 8
// CHECK: %[[m1_ptr:[^ ]*]] = getelementptr inbounds %struct.S, %struct.S* %[[this]], i32 0, i32 0
// CHECK: %[[m1:[^ ]*]] = load i32, i32* %[[m1_ptr]]
// CHECK: %[[l1:[^ ]*]] = load i32, i32* %[[l1_ptr]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 noundef %[[l1]], i32 noundef %[[m1]])

struct V {
  void test_virtual(int p1);
  virtual void virt(int p1);
};

void V::test_virtual(int p1) {
  __try {
    might_crash();
  } __finally {
    virt(p1);
  }
}

// CHECK-LABEL: define dso_local void @"?test_virtual@V@@QEAAXH@Z"(%struct.V* {{[^,]*}} %this, i32 noundef %p1)
// CHECK: @llvm.localescape(%struct.V** %[[this_addr:[^, ]*]], i32* %[[p1_addr:[^, ]*]])
// CHECK: store i32 %p1, i32* %[[p1_addr]], align 4
// CHECK: store %struct.V* %this, %struct.V** %[[this_addr]], align 8
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal void @"?fin$0@0@test_virtual@V@@"(i8 noundef %abnormal_termination, i8* noundef %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%struct.V*, i32)* @"?test_virtual@V@@QEAAXH@Z" to i8*), i8* %frame_pointer, i32 0)
// CHECK: %[[this_ptr:[^ ]*]] = bitcast i8* %[[this_i8]] to %struct.V**
// CHECK: %[[this:[^ ]*]] = load %struct.V*, %struct.V** %[[this_ptr]], align 8
// CHECK: %[[p1_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%struct.V*, i32)* @"?test_virtual@V@@QEAAXH@Z" to i8*), i8* %frame_pointer, i32 1)
// CHECK: %[[p1_ptr:[^ ]*]] = bitcast i8* %[[p1_i8]] to i32*
// CHECK: %[[p1:[^ ]*]] = load i32, i32* %[[p1_ptr]]
// CHECK: %[[this_2:[^ ]*]] = bitcast %struct.V* %[[this]] to void (%struct.V*, i32)***
// CHECK: %[[vtable:[^ ]*]] = load void (%struct.V*, i32)**, void (%struct.V*, i32)*** %[[this_2]], align 8
// CHECK: %[[vfn:[^ ]*]] = getelementptr inbounds void (%struct.V*, i32)*, void (%struct.V*, i32)** %[[vtable]], i64 0
// CHECK: %[[virt:[^ ]*]] = load void (%struct.V*, i32)*, void (%struct.V*, i32)** %[[vfn]], align 8
// CHECK: call void %[[virt]](%struct.V* {{[^,]*}} %[[this]], i32 noundef %[[p1]])

void test_lambda() {
  int l1 = 13;
  auto lambda = [&]() {
    int l2 = 42;
    __try {
      might_crash();
    } __except (basic_filter(l1, l2)) {
    }
  };
  lambda();
}

// CHECK-LABEL: define internal void @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ"(%class.anon* {{[^,]*}} %this)
// CHECK: @llvm.localescape(%class.anon** %[[this_addr:[^, ]*]], i32* %[[l2_addr:[^, ]*]])
// CHECK: store %class.anon* %this, %class.anon** %[[this_addr]], align 8
// CHECK: store i32 42, i32* %[[l2_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@?R<lambda_0>@?0??test_lambda@@YAXXZ@"(i8* noundef %exception_pointers, i8* noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.eh.recoverfp(i8* bitcast (void (%class.anon*)* @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ" to i8*), i8* %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%class.anon*)* @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ" to i8*), i8* %[[fp]], i32 0)
// CHECK: %[[this_ptr:[^ ]*]] = bitcast i8* %[[this_i8]] to %class.anon**
// CHECK: %[[this:[^ ]*]] = load %class.anon*, %class.anon** %[[this_ptr]], align 8
// CHECK: %[[l2_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%class.anon*)* @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ" to i8*), i8* %[[fp]], i32 1)
// CHECK: %[[l2_ptr:[^ ]*]] = bitcast i8* %[[l2_i8]] to i32*
// CHECK: %[[l2:[^ ]*]] = load i32, i32* %[[l2_ptr]]
// CHECK: %[[l1_ref_ptr:[^ ]*]] = getelementptr inbounds %class.anon, %class.anon* %[[this]], i32 0, i32 0
// CHECK: %[[l1_ref:[^ ]*]] = load i32*, i32** %[[l1_ref_ptr]]
// CHECK: %[[l1:[^ ]*]] = load i32, i32* %[[l1_ref]]
// CHECK: call i32 (i32, ...) @basic_filter(i32 noundef %[[l1]], i32 noundef %[[l2]])

struct U {
  void this_in_lambda();
};

void U::this_in_lambda() {
  auto lambda = [=]() {
    __try {
      might_crash();
    } __except (basic_filter(0, this)) {
    }
  };
  lambda();
}

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@?R<lambda_1>@?0??this_in_lambda@U@@QEAAXXZ@"(i8* noundef %exception_pointers, i8* noundef %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call i8* @llvm.localrecover(i8* bitcast (void (%class.anon.0*)* @"??R<lambda_1>@?0??this_in_lambda@U@@QEAAXXZ@QEBA@XZ" to i8*), i8* %[[fp:[^ ]*]], i32 0)
// CHECK: %[[this_ptr:[^ ]*]] = bitcast i8* %[[this_i8]] to %class.anon.0**
// CHECK: %[[this:[^ ]*]] = load %class.anon.0*, %class.anon.0** %[[this_ptr]], align 8
// CHECK: %[[actual_this_ptr:[^ ]*]] = getelementptr inbounds %class.anon.0, %class.anon.0* %[[this]], i32 0, i32 0
// CHECK: %[[actual_this:[^ ]*]] = load %struct.U*, %struct.U** %[[actual_this_ptr]], align 8
// CHECK: call i32 (i32, ...) @basic_filter(i32 noundef 0, %struct.U* noundef %[[actual_this]])
