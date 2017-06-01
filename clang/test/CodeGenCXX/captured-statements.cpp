// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -emit-llvm %s -o %t
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-1
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-2
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-3
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-4
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-5
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-6
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK-7

struct Foo {
  int x;
  float y;
  ~Foo() {}
};

struct TestClass {
  int x;

  TestClass() : x(0) {};
  void MemberFunc() {
    Foo f;
    #pragma clang __debug captured
    {
      static double inner = x;
      (void)inner;
      f.y = x;
    }
  }
};

void test1() {
  TestClass c;
  c.MemberFunc();
  // CHECK-1: %[[Capture:struct\.anon[\.0-9]*]] = type { %struct.TestClass*, %struct.Foo* }
  // CHECK-1: [[INNER:@.+]] = {{.+}} global double

  // CHECK-1: define {{.*}} void @_ZN9TestClass10MemberFuncEv
  // CHECK-1:   alloca %struct.anon
  // CHECK-1:   getelementptr inbounds %[[Capture]], %[[Capture]]* %{{[^,]*}}, i32 0, i32 0
  // CHECK-1:   getelementptr inbounds %[[Capture]], %[[Capture]]* %{{[^,]*}}, i32 0, i32 1
  // CHECK-1:   store %struct.Foo* %f, %struct.Foo**
  // CHECK-1:   call void @[[HelperName:[\.A-Za-z0-9_]+]](%[[Capture]]*
  // CHECK-1:   call {{.*}}FooD1Ev
  // CHECK-1:   ret
}

// CHECK-1: define internal {{.*}}void @[[HelperName]]
// CHECK-1:   getelementptr inbounds %[[Capture]], %[[Capture]]* {{[^,]*}}, i32 0, i32 0
// CHECK-1:   call {{.*}}i32 @__cxa_guard_acquire(
// CHECK-1:   store double %{{.+}}, double* [[INNER]],
// CHECK-1:   call {{.*}}void @__cxa_guard_release(
// CHECK-1:   getelementptr inbounds %struct.TestClass, %struct.TestClass* {{[^,]*}}, i32 0, i32 0
// CHECK-1:   getelementptr inbounds %[[Capture]], %[[Capture]]* {{[^,]*}}, i32 0, i32 1

void test2(int x) {
  int y = [&]() {
    #pragma clang __debug captured
    {
      x++;
    }
    return x;
  }();

  // CHECK-2-LABEL: define {{.*}}void @_Z5test2i
  // CHECK-2:   call {{.*}} @[[Lambda:["$\w]+]]
  //
  // CHECK-2: define internal {{.*}} @[[Lambda]]
  // CHECK-2:   call void @[[HelperName:["$_A-Za-z0-9]+]](%[[Capture:.*]]*
  //
  // CHECK-2: define internal {{.*}}void @[[HelperName]]
  // CHECK-2:   getelementptr inbounds %[[Capture]], %[[Capture]]*
  // CHECK-2:   load i32*, i32**
  // CHECK-2:   load i32, i32*
}

void test3(int x) {
  #pragma clang __debug captured
  {
    x = [=]() { return x + 1; } ();
  }
  x = [=]() { return x + 1; }();

  // CHECK-3: %[[Capture:struct\.anon[\.0-9]*]] = type { i32* }

  // CHECK-3-LABEL: define {{.*}}void @_Z5test3i
  // CHECK-3:   store i32*
  // CHECK-3:   call void @{{.*}}__captured_stmt
  // CHECK-3:   ret void
}

void test4() {
  #pragma clang __debug captured
  {
    Foo f;
    f.x = 5;
  }
  // CHECK-4-LABEL: define {{.*}}void @_Z5test4v
  // CHECK-4:   call void @[[HelperName:[\."$_A-Za-z0-9]+]](%[[Capture:.*]]*
  // CHECK-4:   ret void
  //
  // CHECK-4: define internal {{.*}}void @[[HelperName]]
  // CHECK-4:   store i32 5, i32*
  // CHECK-4:   call {{.*}}FooD1Ev
}

template <typename T, int id>
void touch(const T &) {}

template <typename T, unsigned id>
void template_capture_var() {
  T x;
  #pragma clang __debug captured
  {
    touch<T, id>(x);
  }
}

template <typename T, int id>
class Val {
  T v;
public:
  void set() {
    #pragma clang __debug captured
    {
      touch<T, id>(v);
    }
  }

  template <typename U, int id2>
  void foo(U u) {
    #pragma clang __debug captured
    {
      touch<U, id + id2>(u);
    }
  }
};

void test_capture_var() {
  // CHECK-5: define {{.*}} void @_Z20template_capture_varIiLj201EEvv
  // CHECK-5-NOT: }
  // CHECK-5: store i32*
  // CHECK-5: call void @__captured_stmt
  // CHECK-5-NEXT: ret void
  template_capture_var<int, 201>();

  // CHECK-5: define {{.*}} void @_ZN3ValIfLi202EE3setEv
  // CHECK-5-NOT: }
  // CHECK-5: store %class.Val*
  // CHECK-5: call void @__captured_stmt
  // CHECK-5-NEXT: ret void
  Val<float, 202> Obj;
  Obj.set();

  // CHECK-5: define {{.*}} void @_ZN3ValIfLi202EE3fooIdLi203EEEvT_
  // CHECK-5-NOT: }
  // CHECK-5: store %class.Val*
  // CHECK-5: store double
  // CHECK-5: call void @__captured_stmt
  // CHECK-5-NEXT: ret void
  Obj.foo<double, 203>(1.0);
}

template <typename T>
void template_capture_lambda() {
  T x, y;
  [=, &y]() {
    #pragma clang __debug captured
    {
      y += x;
    }
  }();
}

void test_capture_lambda() {
  // CHECK-6: define {{.*}} void @_ZZ23template_capture_lambdaIiEvvENKUlvE_clEv
  // CHECK-6-NOT: }
  // CHECK-6: store i32*
  // CHECK-6: store i32*
  // CHECK-6: call void @__captured_stmt
  // CHECK-6-NEXT: ret void
  template_capture_lambda<int>();
}

inline int test_captured_linkage() {
  // CHECK-7: @_ZZ21test_captured_linkagevE1i = linkonce_odr global i32 0
  int j;
  #pragma clang __debug captured
  {
    static int i = 0;
    j = ++i;
  }
  return j;
}
void call_test_captured_linkage() {
  test_captured_linkage();
}
