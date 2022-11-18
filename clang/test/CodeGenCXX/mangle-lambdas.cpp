// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s -w | FileCheck %s

void side_effect();

// CHECK-LABEL: define linkonce_odr void @_Z11inline_funci
inline void inline_func(int n) {
  // CHECK: call noundef i32 @_ZZ11inline_funciENKUlvE_clEv
  int i = []{ return side_effect(), 1; }();

  // CHECK: call noundef i32 @_ZZ11inline_funciENKUlvE0_clEv
  int j = [=] { return n + i; }();

  // CHECK: call noundef double @_ZZ11inline_funciENKUlvE1_clEv
  int k = [=] () -> double { return n + i; }();

  // CHECK: call noundef i32 @_ZZ11inline_funciENKUliE_clEi
  int l = [=] (int x) -> int { return x + i; }(n);

  int inner(int i = []{ return side_effect(), 17; }());
  // CHECK: call noundef i32 @_ZZ11inline_funciENKUlvE2_clEv
  // CHECK-NEXT: call noundef i32 @_Z5inneri
  inner();

  // CHECK-NEXT: ret void
}

void call_inline_func() {
  inline_func(17);
}

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZNK10inline_varMUlvE_clEv(
// CHECK: @_ZZNK10inline_varMUlvE_clEvE1n
inline auto inline_var = [] {
  static int n = 5;
  return &n;
};

int *use_inline_var = inline_var();

// CHECK-LABEL: define linkonce_odr noundef ptr @_ZNK12var_templateIiEMUlvE_clEv(
// CHECK: @_ZZNK12var_templateIiEMUlvE_clEvE1n
template<typename T> auto var_template = [] {
  static int n = 9;
  return &n;
};

int *use_var_template = var_template<int>();

// CHECK-LABEL: define {{.*}} @_Z29use_var_template_substitutionN12var_templateIiEMUlvE_ENS_IfEMUlvE_E
void use_var_template_substitution(decltype(var_template<int>), decltype(var_template<float>)) {}

struct S {
  void f(int = []{return side_effect(), 1;}()
             + []{return side_effect(), 2;}(),
         int = []{return side_effect(), 3;}());
  void g(int, int);
};

void S::g(int i = []{return side_effect(), 1;}(),
          int j = []{return side_effect(), 2; }()) {}

// CHECK-LABEL: define{{.*}} void @_Z6test_S1S
void test_S(S s) {
  // CHECK: call noundef i32 @_ZZN1S1fEiiEd0_NKUlvE_clEv
  // CHECK-NEXT: call noundef i32 @_ZZN1S1fEiiEd0_NKUlvE0_clEv
  // CHECK-NEXT: add nsw i32
  // CHECK-NEXT: call noundef i32 @_ZZN1S1fEiiEd_NKUlvE_clEv
  // CHECK-NEXT: call void @_ZN1S1fEii
  s.f();

  // NOTE: These manglings don't actually matter that much, because
  // the lambdas in the default arguments of g() won't be seen by
  // multiple translation units. We check them mainly to ensure that they don't 
  // get the special mangling for lambdas in in-class default arguments.
  // CHECK: call noundef i32 @"_ZNK1S3$_0clEv"
  // CHECK-NEXT: call noundef i32 @"_ZNK1S3$_1clEv"
  // CHECK-NEXT: call void @_ZN1S1gEi
  s.g();

  // CHECK-NEXT: ret void
}

// Check the linkage of the lambda call operators used in test_S.
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN1S1fEiiEd0_NKUlvE_clEv
// CHECK: ret i32 1
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN1S1fEiiEd0_NKUlvE0_clEv
// CHECK: ret i32 2
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN1S1fEiiEd_NKUlvE_clEv
// CHECK: ret i32 3
// CHECK-LABEL: define internal noundef i32 @"_ZNK1S3$_0clEv"
// CHECK: ret i32 1
// CHECK-LABEL: define internal noundef i32 @"_ZNK1S3$_1clEv"
// CHECK: ret i32 2

template<typename T>
struct ST {
  void f(T = []{return T() + 1;}()
           + []{return T() + 2;}(),
         T = []{return T(3);}());
};

// CHECK-LABEL: define{{.*}} void @_Z7test_ST2STIdE
void test_ST(ST<double> st) {
  // CHECK: call noundef double @_ZZN2STIdE1fEddEd0_NKUlvE_clEv
  // CHECK-NEXT: call noundef double @_ZZN2STIdE1fEddEd0_NKUlvE0_clEv
  // CHECK-NEXT: fadd double
  // CHECK-NEXT: call noundef double @_ZZN2STIdE1fEddEd_NKUlvE_clEv
  // CHECK-NEXT: call void @_ZN2STIdE1fEdd
  st.f();

  // CHECK-NEXT: ret void
}

// Check the linkage of the lambda call operators used in test_ST.
// CHECK-LABEL: define linkonce_odr noundef double @_ZZN2STIdE1fEddEd0_NKUlvE_clEv
// CHECK: ret double 1
// CHECK-LABEL: define linkonce_odr noundef double @_ZZN2STIdE1fEddEd0_NKUlvE0_clEv
// CHECK: ret double 2
// CHECK-LABEL: define linkonce_odr noundef double @_ZZN2STIdE1fEddEd_NKUlvE_clEv
// CHECK: ret double 3

template<typename T> 
struct StaticMembers {
  static T x;
  static T y;
  static T z;
  static int (*f)();
};

template<typename T> int accept_lambda(T);

template<typename T>
T StaticMembers<T>::x = []{return side_effect(), 1;}() + []{return side_effect(), 2;}();

template<typename T>
T StaticMembers<T>::y = []{return side_effect(), 3;}();

template<typename T>
T StaticMembers<T>::z = accept_lambda([]{return side_effect(), 4;});

template<typename T>
int (*StaticMembers<T>::f)() = (side_effect(), []{return side_effect(), 5;});

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call noundef i32 @_ZNK13StaticMembersIfE1xMUlvE_clEv
// CHECK-NEXT: call noundef i32 @_ZNK13StaticMembersIfE1xMUlvE0_clEv
// CHECK-NEXT: add nsw
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK13StaticMembersIfE1xMUlvE_clEv
// CHECK: ret i32 1
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK13StaticMembersIfE1xMUlvE0_clEv
// CHECK: ret i32 2
template float StaticMembers<float>::x;

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call noundef i32 @_ZNK13StaticMembersIfE1yMUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK13StaticMembersIfE1yMUlvE_clEv
// CHECK: ret i32 3
template float StaticMembers<float>::y;

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call noundef i32 @_Z13accept_lambdaIN13StaticMembersIfE1zMUlvE_EEiT_
// CHECK: declare noundef i32 @_Z13accept_lambdaIN13StaticMembersIfE1zMUlvE_EEiT_()
template float StaticMembers<float>::z;

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call {{.*}} @_ZNK13StaticMembersIfE1fMUlvE_cvPFivEEv
// CHECK-LABEL: define linkonce_odr noundef ptr @_ZNK13StaticMembersIfE1fMUlvE_cvPFivEEv
template int (*StaticMembers<float>::f)();

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: call noundef i32 @"_ZNK13StaticMembersIdE3$_2clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZNK13StaticMembersIdE3$_2clEv"
// CHECK: ret i32 42
template<> double StaticMembers<double>::z = []{return side_effect(), 42; }();

template<typename T>
void func_template(T = []{ return T(); }());

// CHECK-LABEL: define{{.*}} void @_Z17use_func_templatev()
void use_func_template() {
  // CHECK: call noundef i32 @"_ZZ13func_templateIiEvT_ENK3$_0clEv"
  func_template<int>();
}

namespace std {
  struct type_info {
    bool before(const type_info &) const noexcept;
  };
}
namespace PR12123 {
  struct A { virtual ~A(); } g;
  struct C { virtual ~C(); } k;
  struct B {
    void f(const std::type_info& x = typeid([]()->A& { return g; }()));
    void h();
    void j(bool cond = typeid([]() -> A & { return g; }()).before(typeid([]() -> C & { return k; }())));
  };
  void B::h() { f(); j(); }
}

// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZZN7PR121231B1fERKSt9type_infoEd_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZZN7PR121231B1jEbEd_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZZN7PR121231B1jEbEd_NKUlvE0_clEv

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]*}}testVarargsLambdaNumberingv(
inline int testVarargsLambdaNumbering() {
  // CHECK: testVarargsLambdaNumberingvE{{.*}}UlzE_
  auto a = [](...) { static int n; return ++n; };
  // CHECK: testVarargsLambdaNumberingvE{{.*}}UlvE_
  auto b = []() { static int n; return ++n; };
  return a() + b();
}
int k = testVarargsLambdaNumbering();


template<typename = int>
void ft1(int = [](int p = [] { return side_effect(), 42; } ()) {
                 return p;
               } ());
void test_ft1() {
  // CHECK: call noundef i32 @"_ZZZ3ft1IiEviENK3$_0clEiEd_NKUlvE_clEv"
  // CHECK: call noundef i32 @"_ZZ3ft1IiEviENK3$_0clEi"
  ft1();
}
// CHECK-LABEL: define internal noundef i32 @"_ZZ3ft1IiEviENK3$_0clEi"
// CHECK-LABEL: define internal noundef i32 @"_ZZZ3ft1IiEviENK3$_0clEiEd_NKUlvE_clEv"

struct c1 {
  template<typename = int>
  void mft1(int = [](int p = [] { return side_effect(), 42; } ()) {
                    return p;
                  } ());
};
void test_c1_mft1() {
  // CHECK: call noundef i32 @_ZZZN2c14mft1IiEEviEd_NKUliE_clEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZN2c14mft1IiEEviEd_NKUliE_clEi
  c1{}.mft1();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN2c14mft1IiEEviEd_NKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZN2c14mft1IiEEviEd_NKUliE_clEiEd_NKUlvE_clEv

template<typename = int>
struct ct1 {
  void mf1(int = [](int p = [] { return side_effect(), 42; } ()) {
                   return p;
                 } ());
  friend void ff(ct1, int = [](int p = [] { return side_effect(), 0; }()) { return p; }()) {}
};
void test_ct1_mft1() {
  // CHECK: call noundef i32 @_ZZZN3ct1IiE3mf1EiEd_NKUliE_clEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZN3ct1IiE3mf1EiEd_NKUliE_clEi
  ct1<>{}.mf1();
  // CHECK: call noundef i32 @_ZZZ2ff3ct1IiEiEd_NKUliE_clEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZ2ff3ct1IiEiEd_NKUliE_clEi
  ff(ct1<>{});
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZN3ct1IiE3mf1EiEd_NKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZN3ct1IiE3mf1EiEd_NKUliE_clEiEd_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ2ff3ct1IiEiEd_NKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ2ff3ct1IiEiEd_NKUliE_clEiEd_NKUlvE_clEv

template<typename = int>
void ft2() {
  [](int p = [] { return side_effect(), 42; } ()) { return p; } ();
}
template void ft2<>();
// CHECK: call noundef i32 @_ZZZ3ft2IiEvvENKUliE_clEiEd_NKUlvE_clEv
// CHECK: call noundef i32 @_ZZ3ft2IiEvvENKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ3ft2IiEvvENKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft2IiEvvENKUliE_clEiEd_NKUlvE_clEv

template<typename>
void ft3() {
  void f(int = []{ return side_effect(), 0; }());
  f();
}
template void ft3<int>();
// CHECK: call noundef i32 @"_ZZ1fiENK3$_0clEv"
// CHECK-LABEL: define internal noundef i32 @"_ZZ1fiENK3$_0clEv"

template<typename>
void ft4() {
  struct lc {
    void mf(int = []{ return side_effect(), 0; }()) {}
  };
  lc().mf();
}
template void ft4<int>();
// CHECK: call noundef i32 @_ZZZ3ft4IiEvvEN2lc2mfEiEd_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft4IiEvvEN2lc2mfEiEd_NKUlvE_clEv


// Check linkage of the various lambdas.
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ11inline_funciENKUlvE_clEv
// CHECK: ret i32 1
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ11inline_funciENKUlvE0_clEv
// CHECK: ret i32
// CHECK-LABEL: define linkonce_odr noundef double @_ZZ11inline_funciENKUlvE1_clEv
// CHECK: ret double
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ11inline_funciENKUliE_clEi
// CHECK: ret i32
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ11inline_funciENKUlvE2_clEv
// CHECK: ret i32 17

// CHECK-LABEL: define linkonce_odr void @_ZN7MembersC2Ev
// CHECK: call noundef i32 @_ZNK7Members1xMUlvE_clEv
// CHECK-NEXT: call noundef i32 @_ZNK7Members1xMUlvE0_clE
// CHECK-NEXT: add nsw i32
// CHECK: call noundef i32 @_ZNK7Members1yMUlvE_clEv
// CHECK: ret void


// Check the linkage of the lambdas used in test_Members.
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK7Members1xMUlvE_clEv
// CHECK: ret i32 1
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK7Members1xMUlvE0_clEv
// CHECK: ret i32 2
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK7Members1yMUlvE_clEv
// CHECK: ret i32 3

// CHECK-LABEL: define linkonce_odr void @_Z1fIZZNK23TestNestedInstantiationclEvENKUlvE_clEvEUlvE_EvT_


namespace PR12808 {
  template <typename> struct B {
    int a;
    template <typename L> constexpr B(L&& x) : a(x()) { }
  };
  template <typename> void b(int) {
    [&]{ (void)B<int>([&]{ return side_effect(), 1; }); }();
  }
  void f() {
    b<int>(1);
  }
  // CHECK-LABEL: define linkonce_odr void @_ZZN7PR128081bIiEEviENKUlvE_clEv
  // CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZN7PR128081bIiEEviENKUlvE_clEvENKUlvE_clEv
}


struct Members {
  int x = [] { return side_effect(), 1; }() + [] { return side_effect(), 2; }();
  int y = [] { return side_effect(), 3; }();
};

void test_Members() {
  Members members;
}

template<typename P> void f(P) { }

struct TestNestedInstantiation {
   void operator()() const {
     []() -> void {
       return f([]{});
     }();
   }
};

void test_NestedInstantiation() {
  TestNestedInstantiation()();
}
