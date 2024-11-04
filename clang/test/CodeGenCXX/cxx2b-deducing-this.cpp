// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-linux -o - | FileCheck %s

struct TrivialStruct {
    void explicit_object_function(this TrivialStruct) {}
};
void test() {
    TrivialStruct s;
    s.explicit_object_function();
}
// CHECK:      define {{.*}}test{{.*}}
// CHECK-NEXT: entry:
// CHECK:      {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:      {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:      call void {{.*}}explicit_object_function{{.*}}
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK:      define {{.*}}explicit_object_function{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:        ret void
// CHECK-NEXT: }


void test_lambda() {
    [](this auto This) -> int {
        return This();
    }();
}

//CHECK: define dso_local void @{{.*}}test_lambda{{.*}}() #0 {
//CHECK: entry:
//CHECK:  %agg.tmp = alloca %class.anon, align 1
//CHECK:  %ref.tmp = alloca %class.anon, align 1
//CHECK:  %call = call noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"()
//CHECK:  ret void
//CHECK: }

//CHECK: define internal noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"() #0 align 2 {
//CHECK: entry:
//CHECK:   %This = alloca %class.anon, align 1
//CHECK:   %agg.tmp = alloca %class.anon, align 1
//CHECK:   %call = call noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"()
//CHECK:   ret i32 %call
//CHECK: }

void test_lambda_ref() {
    auto l = [i = 42](this auto & This, int j) -> int {
        return This(j);
    };
    l(0);
}

// CHECK: define dso_local void @_Z15test_lambda_refv() #0 {
// CHECK: entry:
// CHECK:   %[[This_address:.]] = alloca %class.anon{{.*}}, align 4
// CHECK:   %[[i_addr:.*]] = getelementptr inbounds nuw %class.anon{{.*}}, ptr %[[This_address]], i32 0, i32 0
// CHECK:   store i32 42, ptr %[[i_addr]], align 4
// CHECK:   %call = call noundef i32 @"_ZZ15test_lambda_refvENH3$_0clIS_EEiRT_i"{{.*}}
// CHECK:   ret void
// CHECK: }

// CHECK: define internal noundef i32 @"_ZZ15test_lambda_refvENH3$_0clIS_EEiRT_i"{{.*}}
// CHECK: entry:
// CHECK:  %This.addr = alloca ptr, align 8
// CHECK:  %j.addr = alloca i32, align 4
// CHECK:  store ptr %This, ptr %This.addr, align 8
// CHECK:  store i32 %j, ptr %j.addr, align 4
// CHECK:  %[[this_addr:.*]] = load ptr, ptr %This.addr, align 8
// CHECK:  %[[j_addr:.*]] = load i32, ptr %j.addr, align 4
// CHECK:  %call = call noundef i32 @"_ZZ15test_lambda_refvENH3$_0clIS_EEiRT_i"(ptr noundef nonnull align 4 dereferenceable(4) %[[this_addr]], i32 noundef %[[j_addr]])
// CHECK:  ret i32 %call
// CHECK: }


struct TestPointer {
    void f(this TestPointer &);
};

void test_pointer() {
    TestPointer t;
    using Fn = void(TestPointer&);
    Fn* fn = &TestPointer::f;
    fn(t);
}
//CHECK: define dso_local void @_Z12test_pointerv() #0 {
//CHECK-NEXT: entry:
//CHECK-NEXT:  %t = alloca %struct.TestPointer, align 1
//CHECK-NEXT:  %fn = alloca ptr, align 8
//CHECK-NEXT:  store ptr @_ZNH11TestPointer1fERS_, ptr %fn, align 8
//CHECK:       %[[fn_ptr:.*]] = load ptr, ptr %fn, align 8
//CHECK-NEXT:  call void %[[fn_ptr]](ptr noundef nonnull align 1 dereferenceable(1) %t)
//CHECK-NEXT:  ret void
//CHECK-NEXT: }


struct MaterializedTemporary {
  void foo(this MaterializedTemporary&&);
  MaterializedTemporary();
  ~MaterializedTemporary();
};

void test_temporary() {
  MaterializedTemporary{}.foo();
}

//CHECK: define dso_local void @_Z14test_temporaryv(){{.*}}
//CHECK-NEXT: entry:
//CHECK:    %ref.tmp = alloca %struct.MaterializedTemporary, align 1
//CHECK:    call void @_ZN21MaterializedTemporaryC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp){{.*}}
//CHECK     invoke void @_ZNH21MaterializedTemporary3fooEOS_(ptr noundef nonnull align 1 dereferenceable(1) %ref.tmp){{.*}}

namespace GH86399 {
volatile int a = 0;
struct function {
  function& operator=(function const&) {
    a = 1;
    return *this;
  }
};

void f() {
  function list;

  //CHECK-LABEL: define internal void @"_ZZN7GH863991f{{.*}}"(ptr %{{.*}})
  //CHECK: call {{.*}} @_ZN7GH863998functionaSERKS0_
  //CHECK-NEXT: ret void
  [&list](this auto self) {
    list = function{};
  }();
}
}

namespace GH84163 {
// Just check that this doesn't crash (we were previously not instantiating
// everything that needs instantiating in here).
template <typename> struct S {};

void a() {
  int x;
  const auto l = [&x](this auto&) { S<decltype(x)> q; };
  l();
}
}

namespace GH84425 {
// As above.
void do_thing(int x) {
    auto second = [&](this auto const& self, int b) -> int {
        if (x) return x;
        else return self(x);
    };

     second(1);
}

void do_thing2(int x) {
    auto second = [&](this auto const& self)  {
        if (true) return x;
        else return x;
    };

     second();
}
}

namespace GH79754 {
// As above.
void f() {
  int x;
  [&x](this auto&&) {return x;}();
}
}

namespace GH70604 {
auto dothing(int num)
{
  auto fun =  [&num](this auto&& self) -> void {
    auto copy = num;
  };

  fun();
}
}

namespace GH87210 {
template <typename... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

// CHECK-LABEL: define dso_local void @_ZN7GH872101fEv()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[X:%.*]] = alloca i32
// CHECK-NEXT:    [[Over:%.*]] = alloca %"{{.*}}Overloaded"
// CHECK:         call noundef ptr @"_ZZN7GH872101fEvENH3$_0clINS_10OverloadedIJS0_EEEEEDaRT_"(ptr {{.*}} [[Over]])
void f() {
  int x;
  Overloaded o {
    // CHECK: define internal noundef ptr @"_ZZN7GH872101fEvENH3$_0clINS_10OverloadedIJS0_EEEEEDaRT_"(ptr {{.*}} [[Self:%.*]])
    // CHECK-NEXT:  entry:
    // CHECK-NEXT:    [[SelfAddr:%.*]] = alloca ptr
    // CHECK-NEXT:    store ptr [[Self]], ptr [[SelfAddr]]
    // CHECK-NEXT:    [[SelfPtr:%.*]] = load ptr, ptr [[SelfAddr]]
    // CHECK-NEXT:    [[XRef:%.*]] = getelementptr inbounds nuw %{{.*}}, ptr [[SelfPtr]], i32 0, i32 0
    // CHECK-NEXT:    [[X:%.*]] = load ptr, ptr [[XRef]]
    // CHECK-NEXT:    ret ptr [[X]]
    [&](this auto& self) {
      return &x;
    }
  };
  o();
}

void g() {
  int x;
  Overloaded o {
    [=](this auto& self) {
      return x;
    }
  };
  o();
}
}

namespace GH89541 {
// Same as above; just check that this doesn't crash.
int one = 1;
auto factory(int& x = one) {
  return [&](this auto self) {
    x;
  };
};

using Base = decltype(factory());
struct Derived : Base {
  Derived() : Base(factory()) {}
};

void f() {
  Derived d;
  d();
}
}


namespace P2797 {
struct C {
  void c(this const C&);    // #first
  void c() &;               // #second
  static void c(int = 0);   // #third

  void d() {
    (&C::c)(C{});
    (&C::c)();
  }
};
void test() {
    (void)C{}.d();
}
// CHECK-LABEL: {{.*}} @_ZN5P27971C1dEv
// CHECK: call void @_ZNH5P27971C1cERKS0_
// CHECK: call void @_ZN5P27971C1cEi
}
