// RUN: %clang_cc1 -O0 -cl-std=CL1.2 -triple amdgcn---amdgizcl -emit-llvm %s -o - | FileCheck -check-prefixes=CHECK,CL12 %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn---amdgizcl -emit-llvm %s -o - | FileCheck -check-prefixes=CHECK,CL20 %s

// CL12-LABEL: define void @func1(i32 addrspace(5)* %x)
// CL20-LABEL: define void @func1(i32* %x)
void func1(int *x) {
  // CL12: %[[x_addr:.*]] = alloca i32 addrspace(5)*{{.*}}addrspace(5)
  // CL12: store i32 addrspace(5)* %x, i32 addrspace(5)* addrspace(5)* %[[x_addr]]
  // CL12: %[[r0:.*]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(5)* %[[x_addr]]
  // CL12: store i32 1, i32 addrspace(5)* %[[r0]]
  // CL20: %[[x_addr:.*]] = alloca i32*{{.*}}addrspace(5)
  // CL20: store i32* %x, i32* addrspace(5)* %[[x_addr]]
  // CL20: %[[r0:.*]] = load i32*, i32* addrspace(5)* %[[x_addr]]
  // CL20: store i32 1, i32* %[[r0]]
  *x = 1;
}

// CHECK-LABEL: define void @func2()
void func2(void) {
  // CHECK: %lv1 = alloca i32, align 4, addrspace(5)
  // CHECK: %lv2 = alloca i32, align 4, addrspace(5)
  // CHECK: %la = alloca [100 x i32], align 4, addrspace(5)
  // CL12: %lp1 = alloca i32 addrspace(5)*, align 4, addrspace(5)
  // CL12: %lp2 = alloca i32 addrspace(5)*, align 4, addrspace(5)
  // CL20: %lp1 = alloca i32*, align 8, addrspace(5)
  // CL20: %lp2 = alloca i32*, align 8, addrspace(5)
  // CHECK: %lvc = alloca i32, align 4, addrspace(5)

  // CHECK: store i32 1, i32 addrspace(5)* %lv1
  int lv1;
  lv1 = 1;
  // CHECK: store i32 2, i32 addrspace(5)* %lv2
  int lv2 = 2;

  // CHECK: %[[arrayidx:.*]] = getelementptr inbounds [100 x i32], [100 x i32] addrspace(5)* %la, i64 0, i64 0
  // CHECK: store i32 3, i32 addrspace(5)* %[[arrayidx]], align 4
  int la[100];
  la[0] = 3;

  // CL12: store i32 addrspace(5)* %lv1, i32 addrspace(5)* addrspace(5)* %lp1, align 4
  // CL20: %[[r0:.*]] = addrspacecast i32 addrspace(5)* %lv1 to i32*
  // CL20: store i32* %[[r0]], i32* addrspace(5)* %lp1, align 8
  int *lp1 = &lv1;

  // CHECK: %[[arraydecay:.*]] = getelementptr inbounds [100 x i32], [100 x i32] addrspace(5)* %la, i64 0, i64 0
  // CL12: store i32 addrspace(5)* %[[arraydecay]], i32 addrspace(5)* addrspace(5)* %lp2, align 4
  // CL20: %[[r1:.*]] = addrspacecast i32 addrspace(5)* %[[arraydecay]] to i32*
  // CL20: store i32* %[[r1]], i32* addrspace(5)* %lp2, align 8
  int *lp2 = la;

  // CL12: call void @func1(i32 addrspace(5)* %lv1)
  // CL20: %[[r2:.*]] = addrspacecast i32 addrspace(5)* %lv1 to i32*
  // CL20: call void @func1(i32* %[[r2]])
  func1(&lv1);

  // CHECK: store i32 4, i32 addrspace(5)* %lvc
  // CHECK: store i32 4, i32 addrspace(5)* %lv1
  const int lvc = 4;
  lv1 = lvc;
}

// CHECK-LABEL: define void @func3()
// CHECK: %a = alloca [16 x [1 x float]], align 4, addrspace(5)
// CHECK: %[[CAST:.+]] = bitcast [16 x [1 x float]] addrspace(5)* %a to i8 addrspace(5)*
// CHECK: call void @llvm.memset.p5i8.i64(i8 addrspace(5)* align 4 %[[CAST]], i8 0, i64 64, i1 false)
void func3(void) {
  float a[16][1] = {{0.}};
}
