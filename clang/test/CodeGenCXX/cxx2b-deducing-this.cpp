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
// CHECK:   %[[i_addr:.*]] = getelementptr inbounds %class.anon{{.*}}, ptr %[[This_address]], i32 0, i32 0
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
