// RUN: %clang_cc1 -O2 -triple s390x-linux -emit-llvm -o - %s | FileCheck %s

int test(int x) {
  // CHECK-LABEL: @test
  // CHECK: = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc;
}

int test_assume_boolean_flag(int x) {
  //CHECK-LABEL: @test_assume_boolean_flag
  //CHECK: %0 = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  //CHECK: [[RES:%.*]] = extractvalue { i32, i32 } %0, 1
  //CHECK: %1 = icmp ult i32 [[RES]], 4
  //CHECK: tail call void @llvm.assume(i1 %1)
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc;
}

int test_low_high_transformation(int x) {
  //CHECK-LABEL: @test_low_high_transformation
  //CHECK: %0 = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  //CHECK: [[RES:%.*]] = extractvalue { i32, i32 } %0, 1
  //CHECK: %1 = icmp ult i32 [[RES]], 4
  //CHECK: tail call void @llvm.assume(i1 %1)
  //CHECK: %2 = add nsw i32 [[RES]], -1 
  //CHECK: %3 = icmp ult i32 %2, 2 
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 1 || cc == 2;
}

int test_equal_high_transformation(int x) {
  //CHECK-LABEL: @test_equal_high_transformation
  //CHECK: %0 = tail call { i32, i32 } asm "ahi $0,42\0A", "=d,={@cc},0"(i32 %x)
  //CHECK: [[RES:%.*]] = extractvalue { i32, i32 } %0, 1
  //CHECK: %1 = icmp ult i32 [[RES]], 4
  //CHECK: tail call void @llvm.assume(i1 %1)
  //CHECK: %2 = and i32 [[RES]], 1
  //CHECK: [[RES1:%.*]] = xor i32 %2, 1 
  //CHECK: ret i32 [[RES1]]
  int cc;
  asm ("ahi %[x],42\n" : [x] "+d"(x), "=@cc" (cc));
  return cc == 0 || cc == 2;
}

