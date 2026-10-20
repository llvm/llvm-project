// RUN: %clang_cc1 -triple x86_64-linux -std=c++14 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++17 %s -emit-llvm -o - | FileCheck %s

namespace std {
  template<typename T> class initializer_list {
    const T *data;
    __SIZE_TYPE__ size;

  public:
    initializer_list();
  };
}

namespace ParenBraceInitList {
  struct Vector {
    Vector(std::initializer_list<int>);
    ~Vector();
  };

  struct Base { Base(Vector) {} };

  // CHECK: define {{.*}}18ParenBraceInitList1fILi0EE
  template<int> void f() {
    // CHECK: call {{.*}}18ParenBraceInitList6VectorC1
    // CHECK: call {{.*}}18ParenBraceInitList6VectorD1
    Base({0});
  }
  template void f<0>();
}

namespace StringLiteralInit {
  template <unsigned L> struct V {
    char dep[L] = "foo";
    char nondep[4] = "bar";
    V() {}
  };
  template <unsigned N> void braced() { char buf[sizeof("foo") + N]{"foo"}; }
  void test() {
    V<4> v1;
    V<10> v2;
    braced<0>();
    braced<1>();
  }

  const char *G;
  template <typename T> void f() { T v = "foo"; G = v; }
  template void f<char[]>();
  template void f<const char *>();

  // CHECK-LABEL: define weak_odr void @_ZN17StringLiteralInit1fIA_cEEvv()
  // CHECK: %v = alloca [4 x i8], align 1
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %v, ptr align 1 @__const._ZN17StringLiteralInit1fIA_cEEvv.v, i64 4, i1 false)

  // CHECK-LABEL: define weak_odr void @_ZN17StringLiteralInit1fIPKcEEvv()
  // CHECK-NOT: alloca [4 x i8]
  // CHECK: store ptr @.str, ptr %v, align 8
  // CHECK: ret void

  // CHECK-LABEL: define linkonce_odr void @_ZN17StringLiteralInit6bracedILj0EEEvv()
  // CHECK: %buf = alloca [4 x i8], align 1
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %buf, ptr align 1 @__const._ZN17StringLiteralInit6bracedILj0EEEvv.buf, i64 4, i1 false)

  // CHECK-LABEL: define linkonce_odr void @_ZN17StringLiteralInit6bracedILj1EEEvv()
  // CHECK: %buf = alloca [5 x i8], align 1
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %buf, ptr align 1 @__const._ZN17StringLiteralInit6bracedILj1EEEvv.buf, i64 5, i1 false)

  // CHECK-LABEL: define linkonce_odr void @_ZN17StringLiteralInit1VILj4EEC2Ev(
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %dep, ptr align 1 @.str, i64 4, i1 false)
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %nondep, ptr align 1 [[BAR:@.str.[0-9]+]], i64 4, i1 false)

  // CHECK-LABEL: define linkonce_odr void @_ZN17StringLiteralInit1VILj10EEC2Ev(
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %dep, ptr align 1 @.str.{{[0-9]+}}, i64 10, i1 false)
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %nondep, ptr align 1 [[BAR]], i64 4, i1 false)
}
