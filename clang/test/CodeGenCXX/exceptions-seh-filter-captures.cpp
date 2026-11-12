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
// CHECK: @llvm.localescape(ptr %[[p1_ptr:[^, ]*]], ptr %[[l1_ptr:[^, ]*]])
// CHECK: store i32 %p1, ptr %[[p1_ptr]], align 4
// CHECK: store i32 13, ptr %[[l1_ptr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@test_freefunc@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @test_freefunc, ptr %frame_pointer)
// CHECK: %[[p1_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @test_freefunc, ptr %[[fp]], i32 0)
// CHECK: %[[l1_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @test_freefunc, ptr %[[fp]], i32 1)
// CHECK: %[[s1:[^ ]*]] = load i32, ptr @"?s1@?1??test_freefunc@@9@4HA", align 4
// CHECK: %[[l1:[^ ]*]] = load i32, ptr %[[l1_i8]]
// CHECK: %[[p1:[^ ]*]] = load i32, ptr %[[p1_i8]]
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

// CHECK-LABEL: define dso_local void @"?test_method@S@@QEAAXXZ"(ptr {{[^,]*}} %this)
// CHECK: @llvm.localescape(ptr %[[l1_addr:[^, ]*]], ptr %[[this_addr:[^, ]*]])
// CHECK: store ptr %this, ptr %[[this_addr]], align 8
// CHECK: store i32 13, ptr %[[l1_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@test_method@S@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @"?test_method@S@@QEAAXXZ", ptr %frame_pointer)
// CHECK: %[[l1_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"?test_method@S@@QEAAXXZ", ptr %[[fp]], i32 0)
// CHECK: %[[this_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"?test_method@S@@QEAAXXZ", ptr %[[fp]], i32 1)
// CHECK: %[[this:[^ ]*]] = load ptr, ptr %[[this_i8]], align 8
// CHECK: %[[m1_ptr:[^ ]*]] = getelementptr inbounds nuw %struct.S, ptr %[[this]], i32 0, i32 0
// CHECK: %[[m1:[^ ]*]] = load i32, ptr %[[m1_ptr]]
// CHECK: %[[l1:[^ ]*]] = load i32, ptr %[[l1_i8]]
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

// CHECK-LABEL: define dso_local void @"?test_virtual@V@@QEAAXH@Z"(ptr {{[^,]*}} %this, i32 noundef %p1)
// CHECK: @llvm.localescape(ptr %[[this_addr:[^, ]*]], ptr %[[p1_addr:[^, ]*]])
// CHECK: store i32 %p1, ptr %[[p1_addr]], align 4
// CHECK: store ptr %this, ptr %[[this_addr]], align 8
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal void @"?fin$0@0@test_virtual@V@@"(i8 noundef %abnormal_termination, ptr noundef %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"?test_virtual@V@@QEAAXH@Z", ptr %frame_pointer, i32 0)
// CHECK: %[[this:[^ ]*]] = load ptr, ptr %[[this_i8]], align 8
// CHECK: %[[p1_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"?test_virtual@V@@QEAAXH@Z", ptr %frame_pointer, i32 1)
// CHECK: %[[p1:[^ ]*]] = load i32, ptr %[[p1_i8]]
// CHECK: %[[vtable:[^ ]*]] = load ptr, ptr %[[this]], align 8
// CHECK: %[[vfn:[^ ]*]] = getelementptr inbounds ptr, ptr %[[vtable]], i64 0
// CHECK: %[[virt:[^ ]*]] = load ptr, ptr %[[vfn]], align 8
// CHECK: call void %[[virt]](ptr {{[^,]*}} %[[this]], i32 noundef %[[p1]])

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

// CHECK-LABEL: define internal void @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ"(ptr {{[^,]*}} %this)
// CHECK: @llvm.localescape(ptr %[[this_addr:[^, ]*]], ptr %[[l2_addr:[^, ]*]])
// CHECK: store ptr %this, ptr %[[this_addr]], align 8
// CHECK: store i32 42, ptr %[[l2_addr]], align 4
// CHECK: invoke void @might_crash()

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@?R<lambda_0>@?0??test_lambda@@YAXXZ@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ", ptr %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ", ptr %[[fp]], i32 0)
// CHECK: %[[this:[^ ]*]] = load ptr, ptr %[[this_i8]], align 8
// CHECK: %[[l2_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"??R<lambda_0>@?0??test_lambda@@YAXXZ@QEBA@XZ", ptr %[[fp]], i32 1)
// CHECK: %[[l2:[^ ]*]] = load i32, ptr %[[l2_i8]]
// CHECK: %[[l1_ref_ptr:[^ ]*]] = getelementptr inbounds nuw %class.anon, ptr %[[this]], i32 0, i32 0
// CHECK: %[[l1_ref:[^ ]*]] = load ptr, ptr %[[l1_ref_ptr]]
// CHECK: %[[l1:[^ ]*]] = load i32, ptr %[[l1_ref]]
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

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@?R<lambda_1>@?0??this_in_lambda@U@@QEAAXXZ@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// CHECK: %[[this_i8:[^ ]*]] = call ptr @llvm.localrecover(ptr @"??R<lambda_1>@?0??this_in_lambda@U@@QEAAXXZ@QEBA@XZ", ptr %[[fp:[^ ]*]], i32 0)
// CHECK: %[[this:[^ ]*]] = load ptr, ptr %[[this_i8]], align 8
// CHECK: %[[actual_this_ptr:[^ ]*]] = getelementptr inbounds nuw %class.anon.0, ptr %[[this]], i32 0, i32 0
// CHECK: %[[actual_this:[^ ]*]] = load ptr, ptr %[[actual_this_ptr]], align 8
// CHECK: call i32 (i32, ...) @basic_filter(i32 noundef 0, ptr noundef %[[actual_this]])
