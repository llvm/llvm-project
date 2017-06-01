// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o %t
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-GLOBALS
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-1
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-2
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-3

typedef __INTPTR_TYPE__ intptr_t;

int foo();
int global;

// Single statement
void test1() {
  int i = 0;
  #pragma clang __debug captured
  {
    static float inner = 3.0;
    (void)inner;
    i++;
  }
  // CHECK-1: %struct.anon = type { i32* }
  // CHECK-1: {{.+}} global float 3.0
  //
  // CHECK-1: @test1(
  // CHECK-1: alloca %struct.anon
  // CHECK-1: getelementptr inbounds %struct.anon, %struct.anon*
  // CHECK-1: store i32* %i
  // CHECK-1: call void @[[HelperName:__captured_stmt[\.0-9]+]]
}

// CHECK-1: define internal {{.*}}void @[[HelperName]](%struct.anon
// CHECK-1:   getelementptr inbounds %struct.anon{{.*}}, i32 0, i32 0
// CHECK-1:   load i32*, i32**
// CHECK-1:   load i32, i32*
// CHECK-1:   add nsw i32
// CHECK-1:   store i32

// Compound statement with local variable
void test2(int x) {
  #pragma clang __debug captured
  {
    int i;
    for (i = 0; i < x; i++)
      foo();
  }
  // CHECK-2: @test2(
  // CHECK-2-NOT: %i
  // CHECK-2: call void @[[HelperName:__captured_stmt[\.0-9]+]]
}

// CHECK-2: define internal {{.*}}void @[[HelperName]]
// CHECK-2-NOT: }
// CHECK-2:   %i = alloca i32

// Capture array
void test3(int size) {
  int arr[] = {1, 2, 3, 4, 5};
  int vla_arr[size];
  #pragma clang __debug captured
  {
    arr[2] = vla_arr[size - 1];
  }
  // CHECK-3: @test3(
  // CHECK-3: alloca [5 x i32]
  // CHECK-3: call void @__captured_stmt
}

// Capture VLA array
void test4(intptr_t size, intptr_t vla_arr[size]) {
  #pragma clang __debug captured
  {
    vla_arr[0] = 1;
  }
  // CHECK-3: test4([[INTPTR_T:i.+]] {{.*}}[[SIZE_ARG:%.+]], [[INTPTR_T]]*
  // CHECK-3: store [[INTPTR_T]] {{.*}}[[SIZE_ARG]], [[INTPTR_T]]* [[SIZE_ADDR:%.+]],
  // CHECK-3: [[SIZE:%.+]] = load [[INTPTR_T]], [[INTPTR_T]]* [[SIZE_ADDR]],
  // CHECK-3: [[REF:%.+]] = getelementptr inbounds
  // CHECK-3: store [[INTPTR_T]] [[SIZE]], [[INTPTR_T]]* [[REF]]
  // CHECK-3: call void @__captured_stmt
}

void dont_capture_global() {
  static int s;
  extern int e;
  #pragma clang __debug captured
  {
    global++;
    s++;
    e++;
  }

  // CHECK-GLOBALS: %[[Capture:struct\.anon[\.0-9]*]] = type {}
  // CHECK-GLOBALS: call void @__captured_stmt[[HelperName:[\.0-9]+]](%[[Capture]]
}

// CHECK-GLOBALS: define internal {{.*}}void @__captured_stmt[[HelperName]]
// CHECK-GLOBALS-NOT: ret
// CHECK-GLOBALS:   load i32, i32* @global
// CHECK-GLOBALS:   load i32, i32* @
// CHECK-GLOBALS:   load i32, i32* @e
