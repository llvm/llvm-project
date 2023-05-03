// RUN: %clang_cc1 -S %s -std=c2x -emit-llvm -o - | FileCheck %s

// Test that null <-> nullptr_t conversions work as expected.
typedef typeof(nullptr) nullptr_t;

nullptr_t nullptr_t_val;

void bool_func(bool);
void nullptr_func(nullptr_t);

void test() {
  // Test initialization
  bool bool_from_nullptr_t = nullptr_t_val;
  nullptr_t nullptr_t_from_nullptr = nullptr;
  void *vp_from_nullptr_t = nullptr_t_val;
  nullptr_t nullptr_t_from_vp = (void *)0;
  nullptr_t nullptr_t_from_int = 0;

  // Test assignment
  bool_from_nullptr_t = nullptr_t_val;
  nullptr_t_from_nullptr = nullptr;
  vp_from_nullptr_t = nullptr_t_val;
  nullptr_t_from_vp = (void *)0;
  nullptr_t_from_int = 0;

  // Test calls
  bool_func(nullptr_t_from_nullptr);
  nullptr_func(nullptr_t_from_nullptr);
  nullptr_func(0);
  nullptr_func((void *)0);
  nullptr_func(nullptr);
  nullptr_func(false);

  // Allocation of locals
  // CHECK: %[[bool_from_nullptr_t:.*]] = alloca i8, align 1
  // CHECK: %[[nullptr_t_from_nullptr:.*]] = alloca ptr, align 8
  // CHECK: %[[vp_from_nullptr_t:.*]] = alloca ptr, align 8
  // CHECK: %[[nullptr_t_from_vp:.*]] = alloca ptr, align 8
  // CHECK: %[[nullptr_t_from_int:.*]] = alloca ptr, align 8

  // Initialization of locals
  // CHECK: store i8 0, ptr %[[bool_from_nullptr_t]], align 1
  // CHECK: store ptr null, ptr %[[nullptr_t_from_nullptr]], align 8
  // CHECK: store ptr null, ptr %[[vp_from_nullptr_t]], align 8
  // CHECK: store ptr null, ptr %[[nullptr_t_from_vp]], align 8
  // CHECK: store ptr null, ptr %[[nullptr_t_from_int]], align 8

  // Assignment expressions
  // CHECK: store i8 0, ptr %[[bool_from_nullptr_t]], align 1
  // CHECK: store ptr null, ptr %[[nullptr_t_from_nullptr]], align 8
  // CHECK: store ptr null, ptr %[[vp_from_nullptr_t]], align 8
  // CHECK: store ptr null, ptr %[[nullptr_t_from_vp]], align 8
  // CHECK: store ptr null, ptr %[[nullptr_t_from_int]], align 8

  // Calls
  // CHECK: call void @bool_func(i1 noundef {{zeroext?}} false)
  // CHECK: call void @nullptr_func(ptr null)
  // CHECK: call void @nullptr_func(ptr null)
  // CHECK: call void @nullptr_func(ptr null)
  // CHECK: call void @nullptr_func(ptr null)
  // CHECK: call void @nullptr_func(ptr null)
}

