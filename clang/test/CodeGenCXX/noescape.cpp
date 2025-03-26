// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -fblocks -o - %s | FileCheck %s

struct S {
  int a[4];
  S(int *, int * __attribute__((noescape)));
  S &operator=(int * __attribute__((noescape)));
  void m0(int *, int * __attribute__((noescape)));
  virtual void vm1(int *, int * __attribute__((noescape)));
};

// CHECK: define{{.*}} void @_ZN1SC2EPiS0_(ptr {{.*}}, {{.*}}, {{.*}} noundef captures(none) {{%.*}})
// CHECK: define{{.*}} void @_ZN1SC1EPiS0_(ptr {{.*}}, {{.*}}, {{.*}} noundef captures(none) {{%.*}}) {{.*}} {
// CHECK: call void @_ZN1SC2EPiS0_(ptr {{.*}}, {{.*}}, {{.*}} captures(none) {{.*}})

S::S(int *, int * __attribute__((noescape))) {}

// CHECK: define {{.*}} ptr @_ZN1SaSEPi(ptr {{.*}}, {{.*}} noundef captures(none) {{%.*}})
S &S::operator=(int * __attribute__((noescape))) { return *this; }

// CHECK: define{{.*}} void @_ZN1S2m0EPiS0_(ptr {{.*}}, {{.*}} noundef captures(none) {{%.*}})
void S::m0(int *, int * __attribute__((noescape))) {}

// CHECK: define{{.*}} void @_ZN1S3vm1EPiS0_(ptr {{.*}}, {{.*}} noundef captures(none) {{%.*}})
void S::vm1(int *, int * __attribute__((noescape))) {}

// CHECK-LABEL: define{{.*}} void @_Z5test0P1SPiS1_(
// CHECK: call void @_ZN1SC1EPiS0_(ptr {{.*}}, {{.*}}, {{.*}} noundef captures(none) {{.*}})
// CHECK: call {{.*}} ptr @_ZN1SaSEPi(ptr {{.*}}, {{.*}} noundef captures(none) {{.*}})
// CHECK: call void @_ZN1S2m0EPiS0_(ptr {{.*}}, {{.*}}, {{.*}} noundef captures(none) {{.*}})
// CHECK: call void {{.*}}(ptr {{.*}}, {{.*}}, {{.*}} noundef captures(none) {{.*}})
void test0(S *s, int *p0, int *p1) {
  S t(p0, p1);
  t = p1;
  s->m0(p0, p1);
  s->vm1(p0, p1);
}

namespace std {
  typedef decltype(sizeof(0)) size_t;
}

// CHECK: define {{.*}} @_ZnwmPv({{.*}}, {{.*}} captures(none) {{.*}})
void *operator new(std::size_t, void * __attribute__((noescape)) p) {
  return p;
}

// CHECK-LABEL: define{{.*}} ptr @_Z5test1Pv(
// CHECK: %call = call {{.*}} @_ZnwmPv({{.*}}, {{.*}} captures(none) {{.*}})
void *test1(void *p0) {
  return ::operator new(16, p0);
}

// CHECK-LABEL: define{{.*}} void @_Z5test2PiS_(
// CHECK: call void @"_ZZ5test2PiS_ENK3$_0clES_S_"({{.*}}, {{.*}}, {{.*}} captures(none) {{.*}})
// CHECK: define internal void @"_ZZ5test2PiS_ENK3$_0clES_S_"({{.*}}, {{.*}}, {{.*}} noundef captures(none) {{%.*}})
void test2(int *p0, int *p1) {
  auto t = [](int *, int * __attribute__((noescape))){};
  t(p0, p1);
}

// CHECK-LABEL: define{{.*}} void @_Z5test3PFvU8noescapePiES_(
// CHECK: call void {{.*}}(ptr noundef captures(none) {{.*}})
typedef void (*NoEscapeFunc)(__attribute__((noescape)) int *);

void test3(NoEscapeFunc f, int *p) {
  f(p);
}

namespace TestByref {

struct S {
  S();
  ~S();
  S(const S &);
  int a;
};

typedef void (^BlockTy)(void);
S &getS();
void noescapefunc(__attribute__((noescape)) BlockTy);

// Check that __block variables with reference types are handled correctly.

// CHECK: define{{.*}} void @_ZN9TestByref4testEv(
// CHECK: %[[X:.*]] = alloca ptr, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = load ptr, ptr %[[X]], align 8
// CHECK: store ptr %[[V0]], ptr %[[BLOCK_CAPTURED]], align 8

void test() {
  __block S &x = getS();
  noescapefunc(^{ (void)x; });
}

}
