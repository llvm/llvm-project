// RUN: %clang_cc1 -w -verify -fopenmp -I %S/Inputs -ast-print %s | FileCheck %s --check-prefix=CHECK
// expected-no-diagnostics

static int variable = 100;
static float variable1 = 200;
static float variable2 = variable1;

static int var = 1;

static int var1 = 10;
static int *var2 = &var1;
static int **ptr1 = &var2;

int arr[2] = {1,2};
int (*arrptr)[2] = &arr;

class declare{
  public: int x;
          void print();
};
declare obj1;
declare *obj2 = &obj1;

struct target{
  int x;
  void print();
};
static target S;

#pragma omp declare target
int target_var = variable;
float target_var1 = variable2;
int *ptr = &var;
int ***ptr2 = &ptr1;
int (**ptr3)[2] = &arrptr;
declare **obj3 = &obj2;
target *S1 = &S;
#pragma omp end declare target
// CHECK: #pragma omp declare target
// CHECK-NEXT: static int variable = 100;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static float variable1 = 200;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static float variable2 = variable1;
// CHECK-NEXT: #pragma omp end declare target

// CHECK: #pragma omp declare target
// CHECK-NEXT: static int var = 1;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static int var1 = 10;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static int *var2 = &var1;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static int **ptr1 = &var2;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int arr[2] = {1, 2};
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int (*arrptr)[2] = &arr;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: class declare {
// CHECK-NEXT: public:
// CHECK-NEXT:  int x;
// CHECK-NEXT:  void print();
// CHECK-NEXT: };
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: declare obj1;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: declare *obj2 = &obj1;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: struct target {
// CHECK-NEXT:  int x;
// CHECK-NEXT:  void print();
// CHECK-NEXT: };
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: static target S;
// CHECK-NEXT: #pragma omp end declare target

// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int target_var = variable;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: float target_var1 = variable2;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int *ptr = &var;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int ***ptr2 = &ptr1;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: int (**ptr3)[2] = &arrptr;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: declare **obj3 = &obj2;
// CHECK-NEXT: #pragma omp end declare target
// CHECK-NEXT: #pragma omp declare target
// CHECK-NEXT: target *S1 = &S;
// CHECK-NEXT: #pragma omp end declare target

