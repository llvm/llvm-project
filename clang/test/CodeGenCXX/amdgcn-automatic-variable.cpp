// RUN: %clang_cc1 -O0 -triple amdgcn---amdgiz -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} void @_Z5func1Pi(i32* %x)
void func1(int *x) {
  // CHECK: %[[x_addr:.*]] = alloca i32*{{.*}}addrspace(5)
  // CHECK: %[[r0:.*]] = addrspacecast i32* addrspace(5)* %[[x_addr]] to i32**
  // CHECK: store i32* %x, i32** %[[r0]]
  // CHECK: %[[r1:.*]] = load i32*, i32** %[[r0]]
  // CHECK: store i32 1, i32* %[[r1]]
  *x = 1;
}

// CHECK-LABEL: define{{.*}} void @_Z5func2v()
void func2(void) {
  // CHECK: %lv1 = alloca i32, align 4, addrspace(5)
  // CHECK: %[[r0:.*]] = addrspacecast i32 addrspace(5)* %lv1 to i32*
  // CHECK: %lv2 = alloca i32, align 4, addrspace(5)
  // CHECK: %[[r1:.*]] = addrspacecast i32 addrspace(5)* %lv2 to i32*
  // CHECK: %la = alloca [100 x i32], align 4, addrspace(5)
  // CHECK: %[[r2:.*]] = addrspacecast [100 x i32] addrspace(5)* %la to [100 x i32]*
  // CHECK: %lp1 = alloca i32*, align 8, addrspace(5)
  // CHECK: %[[r3:.*]] = addrspacecast i32* addrspace(5)* %lp1 to i32**
  // CHECK: %lp2 = alloca i32*, align 8, addrspace(5)
  // CHECK: %[[r4:.*]] = addrspacecast i32* addrspace(5)* %lp2 to i32**
  // CHECK: %lvc = alloca i32, align 4, addrspace(5)
  // CHECK: %[[r5:.*]] = addrspacecast i32 addrspace(5)* %lvc to i32*

  // CHECK: store i32 1, i32* %[[r0]]
  int lv1;
  lv1 = 1;
  // CHECK: store i32 2, i32* %[[r1]]
  int lv2 = 2;

  // CHECK: %[[arrayidx:.*]] = getelementptr inbounds [100 x i32], [100 x i32]* %[[r2]], i64 0, i64 0
  // CHECK: store i32 3, i32* %[[arrayidx]], align 4
  int la[100];
  la[0] = 3;

  // CHECK: store i32* %[[r0]], i32** %[[r3]], align 8
  int *lp1 = &lv1;

  // CHECK: %[[arraydecay:.*]] = getelementptr inbounds [100 x i32], [100 x i32]* %[[r2]], i64 0, i64 0
  // CHECK: store i32* %[[arraydecay]], i32** %[[r4]], align 8
  int *lp2 = la;

  // CHECK: call void @_Z5func1Pi(i32* %[[r0]])
  func1(&lv1);

  // CHECK: store i32 4, i32* %[[r5]]
  // CHECK: store i32 4, i32* %[[r0]]
  const int lvc = 4;
  lv1 = lvc;
}

void destroy(int x);

class A {
int x;
public:
  A():x(0) {}
  ~A() {
   destroy(x);
  }
};

// CHECK-LABEL: define{{.*}} void @_Z5func3v
void func3() {
  // CHECK: %[[a:.*]] = alloca %class.A, align 4, addrspace(5)
  // CHECK: %[[r0:.*]] = addrspacecast %class.A addrspace(5)* %[[a]] to %class.A*
  // CHECK: call void @_ZN1AC1Ev(%class.A* {{[^,]*}} %[[r0]])
  // CHECK: call void @_ZN1AD1Ev(%class.A* {{[^,]*}} %[[r0]])
  A a;
}

// CHECK-LABEL: define{{.*}} void @_Z5func4i
void func4(int x) {
  // CHECK: %[[x_addr:.*]] = alloca i32, align 4, addrspace(5)
  // CHECK: %[[r0:.*]] = addrspacecast i32 addrspace(5)* %[[x_addr]] to i32*
  // CHECK: store i32 %x, i32* %[[r0]], align 4
  // CHECK: call void @_Z5func1Pi(i32* %[[r0]])
  func1(&x);
}

// CHECK-LABEL: define{{.*}} void @_Z5func5v
void func5() {
  return;
  int x = 0;
}

// CHECK-LABEL: define{{.*}} void @_Z5func6v
void func6() {
  return;
  int x;
}

// CHECK-LABEL: define{{.*}} void @_Z5func7v
extern void use(int *);
void func7() {
  goto later;
  int x;
later:
  use(&x);
}

// CHECK-NOT: !opencl.ocl.version
