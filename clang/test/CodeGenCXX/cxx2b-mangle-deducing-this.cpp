// RUN: %clang_cc1 -std=c++2b -emit-llvm -triple x86_64-linux -o - %s 2>/dev/null | FileCheck %s

struct S {
friend void test();
public:
    void a(this auto){}
    void b(this auto&){}
    void c(this S){}
    void c(this S, int){}
private:
    void d(this auto){}
    void e(this auto&){}
    void f(this S){}
    void f(this S, int){}
protected:
    void g(this auto){}
    void h(this auto&){}
    void i(this S){}
    void i(this S, int){}
};


void test() {
    S s;
    s.a();
    // CHECK: call void @_ZNH1S1aIS_EEvT_
    s.b();
    // CHECK: call void @_ZNH1S1bIS_EEvRT_
    s.c();
    // CHECK: call void @_ZNH1S1cES_
    s.c(0);
    // CHECK: call void @_ZNH1S1cES_i
    s.d();
    // CHECK: call void @_ZNH1S1dIS_EEvT_
    s.e();
    // CHECK: call void @_ZNH1S1eIS_EEvRT_
    s.f();
    // CHECK: call void @_ZNH1S1fES_
    s.f(0);
    // CHECK: call void @_ZNH1S1fES_i
    s.g();
    // CHECK: call void @_ZNH1S1gIS_EEvT_
    s.h();
    // CHECK: call void @_ZNH1S1hIS_EEvRT_
    s.i();
    // CHECK: call void @_ZNH1S1iES_
    s.i(0);
    // CHECK: call void @_ZNH1S1iES_i
}

struct StaticAndExplicit {
  static void f(StaticAndExplicit);
  void f(this StaticAndExplicit);
};

void test2() {
    StaticAndExplicit s;

    StaticAndExplicit::f(s);
    // CHECK: call void @_ZN17StaticAndExplicit1fES_

    s.f();
    // CHECK: call void @_ZNH17StaticAndExplicit1fES_
}
