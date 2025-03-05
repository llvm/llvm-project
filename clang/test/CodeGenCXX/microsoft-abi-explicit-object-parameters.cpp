// RUN: %clang_cc1 -std=c++2b -emit-llvm -triple=x86_64-pc-win32 -o - %s 2>/dev/null | FileCheck %s

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
    // CHECK: call void @"??$a@US@@@S@@SAX_VU0@@Z"
    s.b();
    // CHECK: call void @"??$b@US@@@S@@SAX_VAEAU0@@Z"
    s.c();
    // CHECK: call void @"?c@S@@SAX_VU1@@Z"
    s.c(0);
    // CHECK: call void @"?c@S@@SAX_VU1@H@Z"
    s.d();
    // CHECK: call void @"??$d@US@@@S@@CAX_VU0@@Z"
    s.e();
    // CHECK: call void @"??$e@US@@@S@@CAX_VAEAU0@@Z"
    s.f();
    // CHECK: call void @"?f@S@@CAX_VU1@@Z"
    s.f(0);
    // CHECK: call void @"?f@S@@CAX_VU1@H@Z"
    s.g();
    // CHECK: call void @"??$g@US@@@S@@KAX_VU0@@Z"
    s.h();
    // CHECK: call void @"??$h@US@@@S@@KAX_VAEAU0@@Z"
    s.i();
    // CHECK: call void @"?i@S@@KAX_VU1@@Z"
    s.i(0);
    // CHECK: call void @"?i@S@@KAX_VU1@H@Z"
}


struct S2 {
  int i = 0;
  void foo(this const S2&, int);
};
struct T {
  S2 bar(this const T&, int);
};
void chain_test() {
  T t;
  t.bar(0).foo(0);
}
// CHECK: define {{.*}}chain_test{{.*}}
// CHECK-NEXT: entry:
// CHECK: {{.*}} = alloca %struct.T, align 1
// CHECK: {{.*}} = alloca %struct.S2, align 4
// CHECK: %call = call i32 @"?bar@T@@SA?AUS2@@_VAEBU1@H@Z"{{.*}}
// CHECK: %coerce.dive = getelementptr inbounds nuw %struct.S2, {{.*}} %{{.*}}, i32 0, i32 0
// CHECK  store i32 %call, ptr %coerce.dive, align 4
// CHECK: call void @"?foo@S2@@SAX_VAEBU1@H@Z"
// CHECK: ret void
