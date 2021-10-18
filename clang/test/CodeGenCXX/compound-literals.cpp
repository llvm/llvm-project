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
// CHECK: @q ={{.*}} global i32* getelementptr inbounds ([5 x i32], [5 x i32]* @.compoundliteral, i32 0, i32 0), align 4

// CHECK-LABEL: define{{.*}} i32 @_Z1fv()
int f() {
  // CHECK: [[LVALUE:%[a-z0-9.]+]] = alloca
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}}, {{.*}}* [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: store i32 17, i32* [[I]]
  // CHECK-NEXT: [[X:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 1
  // CHECK-NEXT: call %struct.X* @_ZN1XC1EPKc({{.*}}[[X]]
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = load i32, i32*
  // CHECK-NEXT: call %struct.Y* @_ZN1YD1Ev
  // CHECK-NEXT: ret i32 [[RESULT]]
  return ((Y){17, "seventeen"}).i;
}

// CHECK-LABEL: define{{.*}} i32 @_Z1gv()
int g() {
  // CHECK: store [2 x i32]* %{{[a-z0-9.]+}}, [2 x i32]** [[V:%[a-z0-9.]+]]
  const int (&v)[2] = (int [2]) {1,2};

  // CHECK: [[A:%[a-z0-9.]+]] = load [2 x i32]*, [2 x i32]** [[V]]
  // CHECK-NEXT: [[A0ADDR:%[a-z0-9.]+]] = getelementptr inbounds [2 x i32], [2 x i32]* [[A]], i32 0, {{.*}} 0
  // CHECK-NEXT: [[A0:%[a-z0-9.]+]] = load i32, i32* [[A0ADDR]]
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
// CHECK: store i32* %{{.*}}, i32** @p

int *q = (int [5]){1, 2, 3, 4, 5};
// (constant initialization, checked above)

extern int n;
int *r = (int [5]){1, 2, 3, 4, 5} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.1()
// CHECK: %[[PTR:.*]] = getelementptr inbounds i32, i32* getelementptr inbounds ([5 x i32], [5 x i32]* @.compoundliteral.2, i32 0, i32 0), i32 %
// CHECK: store i32* %[[PTR]], i32** @r

int *PR21912_1 = (int []){} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.3()
// CHECK: %[[PTR:.*]] = getelementptr inbounds i32, i32* getelementptr inbounds ([0 x i32], [0 x i32]* @.compoundliteral.4, i32 0, i32 0), i32 %
// CHECK: store i32* %[[PTR]], i32** @PR21912_1

union PR21912Ty {
  long long l;
  double d;
};
union PR21912Ty *PR21912_2 = (union PR21912Ty[]){{.d = 2.0}, {.l = 3}} + n;
// CHECK-LABEL: define {{.*}}__cxx_global_var_init.5()
// CHECK: %[[PTR:.*]] = getelementptr inbounds %union.PR21912Ty, %union.PR21912Ty* getelementptr inbounds ([2 x %union.PR21912Ty], [2 x %union.PR21912Ty]* bitcast (<{ { double }, %union.PR21912Ty }>* @.compoundliteral.6 to [2 x %union.PR21912Ty]*), i32 0, i32 0), i32 %
// CHECK: store %union.PR21912Ty* %[[PTR]], %union.PR21912Ty** @PR21912_2, align 4

// This compound literal should have local scope.
int computed_with_lambda = [] {
  int *array = (int[]) { 1, 3, 5, 7 };
  return array[0];
}();
// CHECK-LABEL: define internal i32 @{{.*}}clEv
// CHECK:         alloca [4 x i32]
