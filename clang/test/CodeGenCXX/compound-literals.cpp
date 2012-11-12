// RUN: %clang_cc1 -triple armv7-none-eabi -emit-llvm -o - %s | FileCheck %s

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

// CHECK: define i32 @_Z1fv()
int f() {
  // CHECK: [[LVALUE:%[a-z0-9.]+]] = alloca
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}}* [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: store i32 17, i32* [[I]]
  // CHECK-NEXT: [[X:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 1
  // CHECK-NEXT: call %struct.X* @_ZN1XC1EPKc({{.*}}[[X]]
  // CHECK-NEXT: [[I:%[a-z0-9]+]] = getelementptr inbounds {{.*}} [[LVALUE]], i32 0, i32 0
  // CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = load i32*
  // CHECK-NEXT: call %struct.Y* @_ZN1YD1Ev
  // CHECK-NEXT: ret i32 [[RESULT]]
  return ((Y){17, "seventeen"}).i;
}

// CHECK: define i32 @_Z1gv()
int g() {
  // CHECK: store [2 x i32]* %{{[a-z0-9.]+}}, [2 x i32]** [[V:%[a-z0-9.]+]]
  const int (&v)[2] = (int [2]) {1,2};

  // CHECK: [[A:%[a-z0-9.]+]] = load [2 x i32]** [[V]]
  // CHECK-NEXT: [[A0ADDR:%[a-z0-9.]+]] = getelementptr inbounds [2 x i32]* [[A]], i32 0, {{.*}} 0
  // CHECK-NEXT: [[A0:%[a-z0-9.]+]] = load i32* [[A0ADDR]]
  // CHECK-NEXT: ret i32 [[A0]]
  return v[0];
}

struct Z { int i[3]; };
int *p = (Z){ {1, 2, 3} }.i;
// CHECK: define {{.*}}__cxx_global_var_init()
// CHECK: store i32* getelementptr inbounds (%struct.Z* @.compoundliteral, i32 0, i32 0, i32 0), i32** @p
