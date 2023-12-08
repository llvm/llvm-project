// RUN: %clang_cc1 -std=c++11 -triple armv7-none-eabi -emit-llvm -o - %s | FileCheck %s

struct X {
  X();
  X(const X&);
  X(const char*);
  ~X();
};

struct Y { 
  int i;
  X x;
};

// CHECK: @.compoundliteral = internal global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 4
// CHECK: @q ={{.*}} global ptr @.compoundliteral, align 4

// CHECK-LABEL: define{{.*}} i32 @_Z1fv()
int f() {
  // CHECK: [[LVALUE:%[a-z0-9.]+]] = alloca
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}}, ptr [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: store i32 17, ptr [[I]]
  // CHECK-NEXT: [[X:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 1
  // CHECK-NEXT: call noundef ptr @_ZN1XC1EPKc({{.*}}[[X]]
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = load i32, ptr
  // CHECK-NEXT: call noundef ptr @_ZN1YD1Ev
  // CHECK-NEXT: ret i32 [[RESULT]]
  return ((Y){17, "seventeen"}).i;
}

// CHECK-LABEL: define{{.*}} i32 @_Z1gv()
int g() {
  // CHECK: store ptr %{{[a-z0-9.]+}}, ptr [[V:%[a-z0-9.]+]]
  const int (&v)[2] = (int [2]) {1,2};

  // CHECK: [[A:%[a-z0-9.]+]] = load ptr, ptr [[V]]
  // CHECK-NEXT: [[A0ADDR:%[a-z0-9.]+]] = getelementptr inbounds [2 x i32], ptr [[A]], i32 0, {{.*}} 0
  // CHECK-NEXT: [[A0:%[a-z0-9.]+]] = load i32, ptr [[A0ADDR]]
  // CHECK-NEXT: ret i32 [[A0]]
  return v[0];
}

// GCC's compound-literals-in-C++ extension lifetime-extends a compound literal
// (or a C++11 list-initialized temporary!) if:
//  - it is at global scope
//  - it has array type
//  - it has a constant initializer

struct Z { int i[3]; };
int *p = (Z){ {1, 2, 3} }.i;
// CHECK: define {{.*}}__cxx_global_var_init()
// CHECK: alloca %struct.Z
// CHECK: store ptr %{{.*}}, ptr @p

int *q = (int [5]){1, 2, 3, 4, 5};
// (constant initialization, checked above)

extern int n;
int *r = (int [5]){1, 2, 3, 4, 5} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.1()
// CHECK: %[[PTR:.*]] = getelementptr inbounds i32, ptr @.compoundliteral.2, i32 %
// CHECK: store ptr %[[PTR]], ptr @r

int *PR21912_1 = (int []){} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.3()
// CHECK: %[[PTR:.*]] = getelementptr inbounds i32, ptr @.compoundliteral.4, i32 %
// CHECK: store ptr %[[PTR]], ptr @PR21912_1

union PR21912Ty {
  long long l;
  double d;
};
union PR21912Ty *PR21912_2 = (union PR21912Ty[]){{.d = 2.0}, {.l = 3}} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.5()
// CHECK: %[[PTR:.*]] = getelementptr inbounds %union.PR21912Ty, ptr @.compoundliteral.6, i32 %
// CHECK: store ptr %[[PTR]], ptr @PR21912_2, align 4

// This compound literal should have local scope.
int computed_with_lambda = [] {
  int *array = (int[]) { 1, 3, 5, 7 };
  return array[0];
}();
// CHECK-LABEL: define internal noundef i32 @{{.*}}clEv
// CHECK:         alloca [4 x i32]
