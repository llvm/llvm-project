// RUN: %clang_cc1 %s -O2 -emit-llvm -fextend-this-ptr -o - | FileCheck %s
// Check that we generate a fake_use call with the 'this' pointer as argument.
// The call should appear after the call to bar().

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
// CHECK: call{{.*}}bar
// CHECK: call void (...) @llvm.fake.use(ptr nonnull %this)
    return V;
// CHECK: ret
}
