// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-ASCII -check-prefix=CHECK %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fexec-charset IBM-1047 -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-EBCDIC -check-prefix=CHECK %s

struct S { S(); ~S(); S(const S &); void operator()(int); };
using size_t = decltype(sizeof(int));
S operator"" _x(const char *, size_t);
S operator"" _y(wchar_t);
S operator"" _z(unsigned long long);
S operator"" _f(long double);
S operator"" _r(const char *);
template<char...Cs> S operator"" _t() { return S(); }

// CHECK-ASCII:  @[[s_foo:.*]] = {{.*}} constant [4 x i8] c"foo\00"
// CHECK-EBCDIC: @[[s_foo:.*]] = {{.*}} constant [4 x i8] c"\86\96\96\00"
// CHECK-ASCII:  @[[s_bar:.*]] = {{.*}} constant [4 x i8] c"bar\00"
// CHECK-EBCDIC: @[[s_bar:.*]] = {{.*}} constant [4 x i8] c"\82\81\99\00"
// CHECK-ASCII:  @[[s_123:.*]] = {{.*}} constant [4 x i8] c"123\00"
// CHECK-EBCDIC: @[[s_123:.*]] = {{.*}} constant [4 x i8] c"\F1\F2\F3\00"
// CHECK-ASCII:  @[[s_4_9:.*]] = {{.*}} constant [4 x i8] c"4.9\00"
// CHECK-EBCDIC: @[[s_4_9:.*]] = {{.*}} constant [4 x i8] c"\F4K\F9\00"
// CHECK-ASCII:  @[[s_0xffffeeee:.*]] = {{.*}} constant [11 x i8] c"0xffffeeee\00"
// CHECK-EBCDIC: @[[s_0xffffeeee:.*]] = {{.*}} constant [11 x i8] c"\F0\A7\86\86\86\86\85\85\85\85\00"

void f() {
  // CHECK: call void @_Zli2_xPKcm({{.*}}, ptr noundef @[[s_foo]], i64 noundef 3)
  // CHECK: call void @_Zli2_xPKcm({{.*}}, ptr noundef @[[s_bar]], i64 noundef 3)
  // CHECK: call void @_Zli2_yw({{.*}} 97)
  // CHECK: call void @_Zli2_zy({{.*}} 42)
  // CHECK: call void @_Zli2_fe({{.*}} x86_fp80 noundef 1.000000e+00)
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  "foo"_x, "bar"_x, L'a'_y, 42_z, 1.0_f;

  // CHECK: call void @_Zli2_rPKc({{.*}}, ptr noundef @[[s_123]])
  // CHECK: call void @_Zli2_rPKc({{.*}}, ptr noundef @[[s_4_9]])
  // CHECK: call void @_Zli2_rPKc({{.*}}, ptr noundef @[[s_0xffffeeee]])
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  123_r, 4.9_r, 0xffff\
eeee_r;

  // FIXME: This mangling is insane. Maybe we should have a special case for
  // char parameter packs?
  // CHECK: call void @_Zli2_tIJLc48ELc120ELc49ELc50ELc51ELc52ELc53ELc54ELc55ELc56EEE1Sv({{.*}})
  // CHECK: call void @_ZN1SD1Ev({{.*}})
  0x12345678_t;
}

// CHECK: define {{.*}} @_Zli2_tIJLc48ELc120ELc49ELc50ELc51ELc52ELc53ELc54ELc55ELc56EEE1Sv(

template<typename T> auto g(T t) -> decltype("foo"_x(t)) { return "foo"_x(t); }
template<typename T> auto i(T t) -> decltype(operator"" _x("foo", 3)(t)) { return operator"" _x("foo", 3)(t); }

void h() {
  g(42);
  i(42);
}

// CHECK: define {{.*}} @_Z1hv()
// CHECK:   call void @_Z1gIiEDTclclL_Zli2_xPKcmELA4_S0_ELm3EEfp_EET_(i32 noundef 42)
// CHECK:   call void @_Z1iIiEDTclclL_Zli2_xPKcmELA4_S0_ELi3EEfp_EET_(i32 noundef 42)

// CHECK: define {{.*}} @_Z1gIiEDTclclL_Zli2_xPKcmELA4_S0_ELm3EEfp_EET_(i32
// CHECK:   call void @_Zli2_xPKcm({{.*}}, ptr noundef @{{.*}}, i64 noundef 3)
// CHECK:   call void @_ZN1SclEi
// CHECK:   call void @_ZN1SD1Ev

// CHECK: define {{.*}} @_Z1iIiEDTclclL_Zli2_xPKcmELA4_S0_ELi3EEfp_EET_(i32
// CHECK:   call void @_Zli2_xPKcm({{.*}}, ptr noundef @{{.*}}, i64 noundef 3)
// CHECK:   call void @_ZN1SclEi
// CHECK:   call void @_ZN1SD1Ev
