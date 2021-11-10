// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -fblocks -o - %s | FileCheck %s

struct S {
  int a[4];
  S(int *, int * __attribute__((noescape)));
  S &operator=(int * __attribute__((noescape)));
  void m0(int *, int * __attribute__((noescape)));
  virtual void vm1(int *, int * __attribute__((noescape)));
};

// CHECK: define{{.*}} void @_ZN1SC2EPiS0_(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{%.*}})
// CHECK: define{{.*}} void @_ZN1SC1EPiS0_(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{%.*}}) {{.*}} {
// CHECK: call void @_ZN1SC2EPiS0_(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{.*}})

S::S(int *, int * __attribute__((noescape))) {}

// CHECK: define {{.*}} %struct.S* @_ZN1SaSEPi(%struct.S* {{.*}}, {{.*}} nocapture {{%.*}})
S &S::operator=(int * __attribute__((noescape))) { return *this; }

// CHECK: define{{.*}} void @_ZN1S2m0EPiS0_(%struct.S* {{.*}}, {{.*}} nocapture {{%.*}})
void S::m0(int *, int * __attribute__((noescape))) {}

// CHECK: define{{.*}} void @_ZN1S3vm1EPiS0_(%struct.S* {{.*}}, {{.*}} nocapture {{%.*}})
void S::vm1(int *, int * __attribute__((noescape))) {}

// CHECK-LABEL: define{{.*}} void @_Z5test0P1SPiS1_(
// CHECK: call void @_ZN1SC1EPiS0_(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{.*}})
// CHECK: call {{.*}} %struct.S* @_ZN1SaSEPi(%struct.S* {{.*}}, {{.*}} nocapture {{.*}})
// CHECK: call void @_ZN1S2m0EPiS0_(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{.*}})
// CHECK: call void {{.*}}(%struct.S* {{.*}}, {{.*}}, {{.*}} nocapture {{.*}})
void test0(S *s, int *p0, int *p1) {
  S t(p0, p1);
  t = p1;
  s->m0(p0, p1);
  s->vm1(p0, p1);
}

namespace std {
  typedef decltype(sizeof(0)) size_t;
}

// CHECK: define {{.*}} @_ZnwmPv({{.*}}, {{.*}} nocapture {{.*}})
void *operator new(std::size_t, void * __attribute__((noescape)) p) {
  return p;
}

// CHECK-LABEL: define{{.*}} i8* @_Z5test1Pv(
// CHECK: %call = call {{.*}} @_ZnwmPv({{.*}}, {{.*}} nocapture {{.*}})
void *test1(void *p0) {
  return ::operator new(16, p0);
}

// CHECK-LABEL: define{{.*}} void @_Z5test2PiS_(
// CHECK: call void @"_ZZ5test2PiS_ENK3$_0clES_S_"({{.*}}, {{.*}}, {{.*}} nocapture {{.*}})
// CHECK: define internal void @"_ZZ5test2PiS_ENK3$_0clES_S_"({{.*}}, {{.*}}, {{.*}} nocapture {{%.*}})
void test2(int *p0, int *p1) {
  auto t = [](int *, int * __attribute__((noescape))){};
  t(p0, p1);
}

// CHECK-LABEL: define{{.*}} void @_Z5test3PFvU8noescapePiES_(
// CHECK: call void {{.*}}(i32* nocapture {{.*}})
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
// CHECK: %[[X:.*]] = alloca %[[STRUCT_TESTBYREF:.*]]*, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %{{.*}}*, %[[STRUCT_TESTBYREF]]* }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %{{.*}}*, %[[STRUCT_TESTBYREF]]* }>, <{ i8*, i32, i32, i8*, %{{.*}}*, %[[STRUCT_TESTBYREF]]* }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = load %[[STRUCT_TESTBYREF]]*, %[[STRUCT_TESTBYREF]]** %[[X]], align 8
// CHECK: store %[[STRUCT_TESTBYREF]]* %[[V0]], %[[STRUCT_TESTBYREF]]** %[[BLOCK_CAPTURED]], align 8

void test() {
  __block S &x = getS();
  noescapefunc(^{ (void)x; });
}

}
