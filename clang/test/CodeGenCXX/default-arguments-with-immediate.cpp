// RUN: %clang_cc1 -std=c++2a -triple x86_64-elf-gnu %s -emit-llvm -o - | FileCheck %s

consteval int immediate() { return 0;}
static int ext();
void f(int a = immediate() + ext());

void test_function() {
    f();
    f(0);
    // CHECK: call noundef i32 @_ZL3extv()
    // CHECK: add
    // CHECK: call {{.*}} @_Z1fi
    // CHECK: call {{.*}} @_Z1fi
}

// CHECK: define {{.*}} i32 @_ZL3extv()

static constexpr int not_immediate();
struct A {
    int a = immediate() + not_immediate();
};

void test_member() {
    // CHECK: call void @_ZN1AC2Ev
    A defaulted;
    // CHECK-NOT: call void @_ZN1AC2Ev
    A provided{0};
}

// CHECK: define {{.*}} void @_ZN1AC2Ev{{.*}}
// CHECK: %call = call noundef i32 @_ZL13not_immediatev()

int never_referenced() {return 42;};


namespace not_used {

struct A {
    int a = immediate() + never_referenced();
};
void f(int a = immediate() + never_referenced());

void g() {
    A a{0};
    f(0);
}

}

static int ext() {return 0;}
static constexpr int not_immediate() {return 0;}

// CHECK-NOT: define {{.*}} i32 _ZL16never_referencedv()(
// CHECK: define {{.*}} i32 @_ZL13not_immediatev()
