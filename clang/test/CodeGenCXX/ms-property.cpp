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
  __declspec(property(get=GetY,put=PutY)) int y[][];
  int GetY(int i, int j) { return i+j; }
  void PutY(int i, int j, int k) { j = i = k; }
  __declspec(property(get=GetZ,put=PutZ)) int z[][][];
  int GetZ(int i, int j, int k) { return i+j+k; }
  void PutZ(int i, int j, int k, int v) { j = i = k = v; }
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
  // CHECK: call noundef i32 @"?GetX@S@@QEAAHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 223, i32 noundef 11)
  int j = p1->x[223][11];
  // CHECK: [[J:%.+]] = load i32, ptr %
  // CHECK-NEXT: call void @"?PutX@S@@QEAAXHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 23, i32 noundef 1, i32 noundef [[J]])
  p1->x[23][1] = j;
  // CHECK: call noundef i32 @"?GetY@S@@QEAAHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 123, i32 noundef 22)
  int k = p1->y[123][22];
  // CHECK: [[K:%.+]] = load i32, ptr %
  // CHECK-NEXT: call void @"?PutY@S@@QEAAXHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 16, i32 noundef 2, i32 noundef [[K]])
  p1->y[16][2] = k;
  // CHECK: call noundef i32 @"?GetZ@S@@QEAAHHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 123, i32 noundef 22, i32 noundef 44)
  k = p1->z[123][22][44];
  // CHECK: [[K:%.+]] = load i32, ptr %
  // CHECK-NEXT: call void @"?PutZ@S@@QEAAXHHHH@Z"(ptr {{[^,]*}} %{{.+}}, i32 noundef 16, i32 noundef 2, i32 noundef 32, i32 noundef [[K]])
  p1->z[16][2][32] = k;
  // CHECK: call noundef float @"?GetX@?$St@M@@QEAAMMM@Z"(ptr {{[^,]*}} %{{.+}}, float noundef 2.230000e+02, float noundef 1.100000e+01)
  float j1 = p2->x[223][11];
  // CHECK: [[J1:%.+]] = load float, ptr %
  // CHECK-NEXT: [[CALL:%.+]] = call noundef float @"?PutX@?$St@M@@QEAAMMMM@Z"(ptr {{[^,]*}} %{{.+}}, float noundef 2.300000e+01, float noundef 1.000000e+00, float noundef [[J1]])
  // CHECK-NEXT: [[CONV:%.+]] = fptosi float [[CALL]] to i32
  // CHECK-NEXT: store i32 [[CONV]], ptr
  argc = p2->x[23][1] = j1;
  // CHECK: [[IDX:%.+]] = call noundef i32 @"?idx@@YAHXZ"()
  // CHECK-NEXT: [[CONV:%.+]] = sitofp i32 [[IDX]] to float
  // CHECK-NEXT: [[GET:%.+]] = call noundef float @"?GetX@?$St@M@@QEAAMMM@Z"(ptr {{[^,]*}} %{{.+}}, float noundef [[CONV]], float noundef 1.000000e+00)
  // CHECK-NEXT: [[INC:%.+]] = fadd float [[GET]], 1.000000e+00
  // CHECK-NEXT: [[CONV:%.+]] = sitofp i32 [[IDX]] to float
  // CHECK-NEXT: call noundef float @"?PutX@?$St@M@@QEAAMMMM@Z"(ptr {{[^,]*}} %{{.+}}, float noundef [[CONV]], float noundef 1.000000e+00, float noundef [[INC]])
  ++p2->x[idx()][1];
  // CHECK: call void @"??$foo@H@@YAXHH@Z"(i32 noundef %{{.+}}, i32 noundef %{{.+}})
  foo(argc, (int)argv[0][0]);
  // CHECK: [[P2:%.+]] = load ptr, ptr %
  // CHECK: [[P1:%.+]] = load ptr, ptr %
  // CHECK: [[P1_X_22_33:%.+]] = call noundef i32 @"?GetX@S@@QEAAHHH@Z"(ptr {{[^,]*}} [[P1]], i32 noundef 22, i32 noundef 33)
  // CHECK: [[CAST:%.+]] = sitofp i32 [[P1_X_22_33]] to double
  // CHECK: [[ARGC:%.+]] = load i32, ptr %
  // CHECK: [[T_X:%.+]] = call noundef i32 @"?get_x@Test1@@QEBAHXZ"(ptr {{[^,]*}} %{{.+}})
  // CHECK: [[CAST2:%.+]] = trunc i32 [[T_X]] to i8
  // CHECK: call void @"?PutY@?$St@M@@QEAAXDHN@Z"(ptr {{[^,]*}} [[P2]], i8 noundef [[CAST2]], i32 noundef [[ARGC]], double noundef [[CAST]])
  p2->y[t.X][argc] =  p1->x[22][33];
  // CHECK: [[P2_1:%.+]] = load ptr, ptr
  // CHECK: [[P2_2:%.+]] = load ptr, ptr
  // CHECK: [[P1:%.+]] = load ptr, ptr
  // CHECK: [[ARGC:%.+]] = load i32, ptr %
  // CHECK: [[P1_X_ARGC_0:%.+]] = call noundef i32 @"?GetX@S@@QEAAHHH@Z"(ptr {{[^,]*}} [[P1]], i32 noundef [[ARGC]], i32 noundef 0)
  // CHECK: [[CAST:%.+]] = trunc i32 [[P1_X_ARGC_0]] to i8
  // CHECK: [[P2_Y_p1_X_ARGC_0_T:%.+]] = call noundef i8 @"?GetY@?$St@M@@QEAADDVTest1@@@Z"(ptr {{[^,]*}} [[P2_2]], i8 noundef [[CAST]], ptr noundef %{{.+}})
  // CHECK: [[CAST:%.+]] = sitofp i8 [[P2_Y_p1_X_ARGC_0_T]] to float
  // CHECK: [[J:%.+]] = load i32, ptr %
  // CHECK: [[CAST1:%.+]] = sitofp i32 [[J]] to float
  // CHECK: [[J:%.+]] = load i32, ptr %
  // CHECK: [[CAST2:%.+]] = sitofp i32 [[J]] to float
  // CHECK: call noundef float @"?PutX@?$St@M@@QEAAMMMM@Z"(ptr {{[^,]*}} [[P2_1]], float noundef [[CAST2]], float noundef [[CAST1]], float noundef [[CAST]])
  p2->x[j][j] = p2->y[p1->x[argc][0]][t];
  // CHECK: [[CALL:%.+]] = call noundef ptr @"?GetTest1@Test1@@SAPEAV1@XZ"()
  // CHECK-NEXT: call noundef i32 @"?get_x@Test1@@QEBAHXZ"(ptr {{[^,]*}} [[CALL]])
  return Test1::GetTest1()->X;
}

// CHECK: define linkonce_odr dso_local void @"??$foo@H@@YAXHH@Z"(i32 noundef %{{.+}}, i32 noundef %{{.+}})
// CHECK: call noundef i32 @"?GetX@?$St@H@@QEAAHHH@Z"(ptr {{[^,]*}} [[BAR:%.+]], i32 noundef %{{.+}} i32 noundef %{{.+}})
// CHECK: call noundef i32 @"?PutX@?$St@H@@QEAAHHHH@Z"(ptr {{[^,]*}} [[BAR]], i32 noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}})
// CHECK: call noundef i32 @"?GetX@?$St@H@@QEAAHHH@Z"(ptr {{[^,]*}} [[BAR]], i32 noundef %{{.+}} i32 noundef %{{.+}})
// CHECK: call void @"?PutY@?$St@H@@QEAAXDHN@Z"(ptr {{[^,]*}} [[BAR]], i8 noundef %{{.+}}, i32 noundef %{{.+}}, double noundef %{{.+}}
// CHECK: call noundef i32 @"?GetX@?$St@H@@QEAAHHH@Z"(ptr {{[^,]*}} [[BAR]], i32 noundef %{{.+}} i32 noundef %{{.+}})
// CHECK: call noundef i8 @"?GetY@?$St@H@@QEAADDVTest1@@@Z"(ptr {{[^,]*}} [[BAR]], i8 noundef %{{.+}}, ptr noundef %{{.+}})
// CHECK: call noundef i32 @"?PutX@?$St@H@@QEAAHHHH@Z"(ptr {{[^,]*}} [[BAR]], i32 noundef %{{.+}}, i32 noundef %{{.+}}, i32 noundef %{{.+}})
#endif //HEADER
