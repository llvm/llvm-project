// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness=this -o - | FileCheck %s --implicit-check-not=fake.use
// Check that we generate a fake_use call with the 'this' pointer as argument,
// and no other fake uses.
// The call should appear after the call to bar().

void bar();

class C
{
public:
    bool test(int p);
    C(int v): v(v) {}

private:
    int v;
};

bool C::test(int p)
{
// CHECK-LABEL: define{{.*}}_ZN1C4testEi(ptr{{[^,]*}} %this, i32{{.*}} %p)
// CHECK:   %this.addr = alloca ptr
// CHECK:   store ptr %this, ptr %this.addr
    int res = p - v;

    bar();
// CHECK: call{{.*}}bar

    return res != 0;
// CHECK:      [[FAKE_USE:%.+]] = load ptr, ptr %this.addr
// CHECK-NEXT: call void (...) @llvm.fake.use(ptr{{.*}} [[FAKE_USE]])
// CHECK-NEXT: ret
}

// CHECK: declare void @llvm.fake.use
