// RUN: %clang_cc1 -fms-extensions -emit-llvm -o - %s | FileCheck %s

// CHECK: %struct.test = type { i32, %struct.nested2, i32 }
// CHECK: %struct.nested2 = type { i32, %struct.nested1, i32 }
// CHECK: %struct.nested1 = type { i32, i32 }
typedef struct nested1 {
    int a1;
    int b1;
} NESTED1;

struct nested2 {
    int a;
    NESTED1; 
    int b;
};

struct test {
    int    x;
    struct nested2; 
    int    y;
};


void foo(void)
{
  // CHECK: %var = alloca %struct.test, align 4
  struct test var;

  // CHECK: getelementptr inbounds %struct.test, ptr %var, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.a;

  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %var, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 2
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.b;

  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %var, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested1, ptr %{{.*}}, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.a1;

  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}var, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested1, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.b1;

  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %var, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.x;

  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %var, i32 0, i32 2
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var.y;
}

void foo2(struct test* var)
{
  // CHECK: alloca ptr, align
  // CHECK-NEXT: store ptr %var, ptr %{{.*}}, align
  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->a;

  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 2
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->b;

  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested1, ptr %{{.*}}, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->a1;

  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested2, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: getelementptr inbounds %struct.nested1, ptr %{{.*}}, i32 0, i32 1
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->b1;

  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 0
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->x;

  // CHECK-NEXT: load ptr, ptr %{{.*}}, align
  // CHECK-NEXT: getelementptr inbounds %struct.test, ptr %{{.*}}, i32 0, i32 2
  // CHECK-NEXT: load i32, ptr %{{.*}}, align 4
  var->y;
}
