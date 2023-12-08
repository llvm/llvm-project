// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: invoke void @llvm.seh.scope.end()

// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: %[[src:[0-9-]+]] = load volatile i32, ptr %i
// CHECK-NEXT: invoke void @"?crash@@YAXH@Z"(i32 noundef %[[src]])
// CHECK: invoke void @llvm.seh.try.end()

// ****************************************************************************
// Abstract:     Test CPP unwind Dtoring under SEH -EHa option

void printf(...);
int volatile *NullPtr = 0;
void crash(int i) {
  struct A {
    ~A() {
      printf(" in A dtor \n");
    }
  } ObjA;
  if (i == 0)
    *NullPtr = 0;

  struct B {
    ~B() {
      printf(" in B dtor \n");
    }
  } ObjB;
  if (i == 1)
    *NullPtr = 0;

  struct C {
    ~C() {
      printf(" in C dtor \n");
    }
  } ObjC;
  if (i == 2)
    *NullPtr = 0;
}

#define TRY __try
#define CATCH_ALL __except (1)

int g;
int main() {
  for (int i = 0; i < 3; i++) {
    TRY {
      crash(i);
    }
    CATCH_ALL {
      printf(" Test CPP unwind: in catch handler i = %d \n", i);
    }
  }
  return 0;
}
