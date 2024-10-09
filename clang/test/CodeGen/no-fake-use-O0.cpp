// RUN: %clang_cc1 %s -O0 -emit-llvm -fextend-this-ptr -o -  | FileCheck %s \
// RUN:      --implicit-check-not="call void (...) @llvm.fake.use"
// RUN: %clang_cc1 %s -O0 -emit-llvm -fextend-lifetimes -o - | FileCheck %s \
// RUN:      --implicit-check-not="call void (...) @llvm.fake.use"
// RUN: %clang_cc1 %s -O1 -emit-llvm -fextend-this-ptr -o -  | FileCheck -check-prefix=OPT %s
// RUN: %clang_cc1 %s -O1 -emit-llvm -fextend-lifetimes -o - | FileCheck -check-prefix=OPT %s
// RUN: %clang_cc1 %s -Os -emit-llvm -fextend-this-ptr -o -  | FileCheck -check-prefix=OPT %s
// RUN: %clang_cc1 %s -Os -emit-llvm -fextend-lifetimes -o - | FileCheck -check-prefix=OPT %s
// RUN: %clang_cc1 %s -Oz -emit-llvm -fextend-this-ptr -o -  | FileCheck -check-prefix=OPT %s
// RUN: %clang_cc1 %s -Oz -emit-llvm -fextend-lifetimes -o - | FileCheck -check-prefix=OPT %s
// Check that we do not generate a fake_use call when we are not optimizing. 

extern void bar();

class v
{
public:
    int x;
    int y;
    int z;
    int w;

    v(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
};

class w
{
public:
    v test(int, int, int, int, int, int, int, int, int, int);
    w(int in): a(in), b(1234) {}

private:
    int a;
    int b;
};

v w::test(int q, int w, int e, int r, int t, int y, int u, int i, int o, int p)
{
// CHECK: define{{.*}}test
    int res = q*w + e - r*t + y*u*i*o*p;
    int res2 = (w + e + r + t + y + o)*(p*q);
    int res3 = res + res2;
    int res4 = q*e + t*y*i + p*e*w * 6 * 4 * 3;

    v V(res, res2, res3, res4);

    bar();
// OPT:       call void (...) @llvm.fake.use
    return V;
}
