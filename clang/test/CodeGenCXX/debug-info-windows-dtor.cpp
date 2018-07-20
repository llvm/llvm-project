// RUN: %clang_cc1 -triple i386-unknown-windows-msvc -std=c++11 -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s

struct A {
  virtual ~A() {}
};

struct B {
  virtual ~B() {}
};

template<typename T>
struct AB: A, B {
};

template struct AB<int>;

// CHECK: define {{.*}}@"??_E?$AB@H@@W3AEPAXI@Z"({{.*}} !dbg [[THUNK_VEC_DEL_DTOR:![0-9]*]]
// CHECK: call {{.*}}@"??_G?$AB@H@@UAEPAXI@Z"({{.*}}) #{{[0-9]*}}, !dbg [[THUNK_LOC:![0-9]*]]
// CHECK: define

// CHECK: [[THUNK_VEC_DEL_DTOR]] = distinct !DISubprogram
// CHECK: [[THUNK_LOC]] = !DILocation(line: 0, scope: [[THUNK_VEC_DEL_DTOR]])
