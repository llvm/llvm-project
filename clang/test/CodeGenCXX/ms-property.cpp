// RUN: %clang_cc1 -emit-llvm -triple=x86_64-pc-win32 -fms-compatibility %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fms-compatibility -emit-pch -o %t %s
// RUN: %clang_cc1 -emit-llvm -triple=x86_64-pc-win32 -fms-compatibility -include-pch %t -verify %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

class Test1 {
private:
  int x_;
  double y_;

public:
  Test1(int x) : x_(x) {}
  __declspec(property(get = get_x)) int X;
  int get_x() const { return x_; }
  static Test1 *GetTest1() { return new Test1(10); }
};

class S {
public:
  __declspec(property(get=GetX,put=PutX)) int x[];
  int GetX(int i, int j) { return i+j; }
  void PutX(int i, int j, int k) { j = i = k; }
};

template <typename T>
class St {
public:
  __declspec(property(get=GetX,put=PutX)) T x[];
  T GetX(T i, T j) { return i+j; }
  T GetX() { return 0; }
  T PutX(T i, T j, T k) { return j = i = k; }
  __declspec(property(get=GetY,put=PutY)) T y[];
  char GetY(char i,  Test1 j) { return i+j.get_x(); }
  void PutY(char i, int j, double k) { j = i = k; }
};

template <typename T>
void foo(T i, T j) {
  St<T> bar;
  Test1 t(i);
  bar.x[i][j] = bar.x[i][j];
  bar.y[t.X][j] = bar.x[i][j];
  bar.x[i][j] = bar.y[bar.x[i][j]][t];
}

int idx() { return 7; }

// CHECK-LABEL: main
int main(int argc, char **argv) {
  Test1 t(argc);
  S *p1 = 0;
  St<float> *p2 = 0;
  // CHECK: call i32 @"?GetX@S@@QEAAHHH@Z"(%class.S* {{[^,]*}} %{{.+}}, i32 223, i32 11)
  int j = p1->x[223][11];
  // CHECK: [[J:%.+]] = load i32, i32* %
  // CHECK-NEXT: call void @"?PutX@S@@QEAAXHHH@Z"(%class.S* {{[^,]*}} %{{.+}}, i32 23, i32 1, i32 [[J]])
  p1->x[23][1] = j;
  // CHECK: call float @"?GetX@?$St@M@@QEAAMMM@Z"(%class.St* {{[^,]*}} %{{.+}}, float 2.230000e+02, float 1.100000e+01)
  float j1 = p2->x[223][11];
  // CHECK: [[J1:%.+]] = load float, float* %
  // CHECK-NEXT: [[CALL:%.+]] = call float @"?PutX@?$St@M@@QEAAMMMM@Z"(%class.St* {{[^,]*}} %{{.+}}, float 2.300000e+01, float 1.000000e+00, float [[J1]])
  // CHECK-NEXT: [[CONV:%.+]] = fptosi float [[CALL]] to i32
  // CHECK-NEXT: store i32 [[CONV]], i32*
  argc = p2->x[23][1] = j1;
  // CHECK: [[IDX:%.+]] = call i32 @"?idx@@YAHXZ"()
  // CHECK-NEXT: [[CONV:%.+]] = sitofp i32 [[IDX]] to float
  // CHECK-NEXT: [[GET:%.+]] = call float @"?GetX@?$St@M@@QEAAMMM@Z"(%class.St* {{[^,]*}} %{{.+}}, float [[CONV]], float 1.000000e+00)
  // CHECK-NEXT: [[INC:%.+]] = fadd float [[GET]], 1.000000e+00
  // CHECK-NEXT: [[CONV:%.+]] = sitofp i32 [[IDX]] to float
  // CHECK-NEXT: call float @"?PutX@?$St@M@@QEAAMMMM@Z"(%class.St* {{[^,]*}} %{{.+}}, float [[CONV]], float 1.000000e+00, float [[INC]])
  ++p2->x[idx()][1];
  // CHECK: call void @"??$foo@H@@YAXHH@Z"(i32 %{{.+}}, i32 %{{.+}})
  foo(argc, (int)argv[0][0]);
  // CHECK: [[P2:%.+]] = load %class.St*, %class.St** %
  // CHECK: [[P1:%.+]] = load %class.S*, %class.S** %
  // CHECK: [[P1_X_22_33:%.+]] = call i32 @"?GetX@S@@QEAAHHH@Z"(%class.S* {{[^,]*}} [[P1]], i32 22, i32 33)
  // CHECK: [[CAST:%.+]] = sitofp i32 [[P1_X_22_33]] to double
  // CHECK: [[ARGC:%.+]] = load i32, i32* %
  // CHECK: [[T_X:%.+]] = call i32 @"?get_x@Test1@@QEBAHXZ"(%class.Test1* {{[^,]*}} %{{.+}})
  // CHECK: [[CAST2:%.+]] = trunc i32 [[T_X]] to i8
  // CHECK: call void @"?PutY@?$St@M@@QEAAXDHN@Z"(%class.St* {{[^,]*}} [[P2]], i8 [[CAST2]], i32 [[ARGC]], double [[CAST]])
  p2->y[t.X][argc] =  p1->x[22][33];
  // CHECK: [[P2_1:%.+]] = load %class.St*, %class.St**
  // CHECK: [[P2_2:%.+]] = load %class.St*, %class.St**
  // CHECK: [[P1:%.+]] = load %class.S*, %class.S**
  // CHECK: [[ARGC:%.+]] = load i32, i32* %
  // CHECK: [[P1_X_ARGC_0:%.+]] = call i32 @"?GetX@S@@QEAAHHH@Z"(%class.S* {{[^,]*}} [[P1]], i32 [[ARGC]], i32 0)
  // CHECK: [[CAST:%.+]] = trunc i32 [[P1_X_ARGC_0]] to i8
  // CHECK: [[P2_Y_p1_X_ARGC_0_T:%.+]] = call i8 @"?GetY@?$St@M@@QEAADDVTest1@@@Z"(%class.St* {{[^,]*}} [[P2_2]], i8 [[CAST]], %class.Test1* %{{.+}})
  // CHECK: [[CAST:%.+]] = sitofp i8 [[P2_Y_p1_X_ARGC_0_T]] to float
  // CHECK: [[J:%.+]] = load i32, i32* %
  // CHECK: [[CAST1:%.+]] = sitofp i32 [[J]] to float
  // CHECK: [[J:%.+]] = load i32, i32* %
  // CHECK: [[CAST2:%.+]] = sitofp i32 [[J]] to float
  // CHECK: call float @"?PutX@?$St@M@@QEAAMMMM@Z"(%class.St* {{[^,]*}} [[P2_1]], float [[CAST2]], float [[CAST1]], float [[CAST]])
  p2->x[j][j] = p2->y[p1->x[argc][0]][t];
  // CHECK: [[CALL:%.+]] = call %class.Test1* @"?GetTest1@Test1@@SAPEAV1@XZ"()
  // CHECK-NEXT: call i32 @"?get_x@Test1@@QEBAHXZ"(%class.Test1* {{[^,]*}} [[CALL]])
  return Test1::GetTest1()->X;
}

// CHECK: define linkonce_odr dso_local void @"??$foo@H@@YAXHH@Z"(i32 %{{.+}}, i32 %{{.+}})
// CHECK: call i32 @"?GetX@?$St@H@@QEAAHHH@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR:%.+]], i32 %{{.+}} i32 %{{.+}})
// CHECK: call i32 @"?PutX@?$St@H@@QEAAHHHH@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i32 %{{.+}}, i32 %{{.+}}, i32 %{{.+}})
// CHECK: call i32 @"?GetX@?$St@H@@QEAAHHH@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i32 %{{.+}} i32 %{{.+}})
// CHECK: call void @"?PutY@?$St@H@@QEAAXDHN@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i8 %{{.+}}, i32 %{{.+}}, double %{{.+}}
// CHECK: call i32 @"?GetX@?$St@H@@QEAAHHH@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i32 %{{.+}} i32 %{{.+}})
// CHECK: call i8 @"?GetY@?$St@H@@QEAADDVTest1@@@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i8 %{{.+}}, %class.Test1* %{{.+}})
// CHECK: call i32 @"?PutX@?$St@H@@QEAAHHHH@Z"(%class.St{{.+}}* {{[^,]*}} [[BAR]], i32 %{{.+}}, i32 %{{.+}}, i32 %{{.+}})
#endif //HEADER
