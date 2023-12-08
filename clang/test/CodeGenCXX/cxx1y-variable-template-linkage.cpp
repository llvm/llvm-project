// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++1y -O1 -disable-llvm-passes %s -o - | FileCheck %s -check-prefix=CHECKA -check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++1y -O1 -disable-llvm-passes -fcxx-exceptions %s -o - | FileCheck %s -check-prefix=CHECKB -check-prefix=CHECK
// expected-no-diagnostics

// The variable template specialization x<Foo> generated in each file
// should be 'internal global' and not 'linkonce_odr global'.

template <typename T> int x = 42;
// CHECK-DAG: @_Z1xIiE = linkonce_odr global
// CHECK-DAG: @_Z1xIZL3foovE3FooE = internal global

// 'static' affects the linkage of the global
template <typename T> static int y = 42;
// CHECK-DAG: @_ZL1yIiE = internal global
// CHECK-DAG: @_ZL1yIZL3foovE3FooE = internal global

// 'const' does not
template <typename T> const int z = 42;
// CHECK-DAG: @_Z1zIiE = linkonce_odr constant
// CHECK-DAG: @_Z1zIZL3foovE3FooE = internal constant

template <typename T> T t = 42;
// CHECK-DAG: @_Z1tIiE = linkonce_odr global
// CHECK-DAG: @_Z1tIKiE = linkonce_odr constant

int mode;

// CHECK-DAG: define internal noundef nonnull align 4 dereferenceable(4) ptr @_ZL3foov(
static const int &foo() {
   struct Foo { };

   switch (mode) {
   case 0:
     // CHECK-DAG: @_Z1xIiE
     return x<int>;
   case 1:
     // CHECK-DAG: @_Z1xIZL3foovE3FooE
     return x<Foo>;
   case 2:
     // CHECK-DAG: @_ZL1yIiE
     return y<int>;
   case 3:
     // CHECK-DAG: @_ZL1yIZL3foovE3FooE
     return y<Foo>;
   case 4:
     // CHECK-DAG: @_Z1zIiE
     return z<int>;
   case 5:
     // CHECK-DAG: @_Z1zIZL3foovE3FooE
     return z<Foo>;
   case 6:
     // CHECK-DAG: @_Z1tIiE
     return t<int>;
   case 7:
     // CHECK-DAG: @_Z1tIKiE
     return t<const int>;
   }
}


#if !__has_feature(cxx_exceptions) // File A
// CHECKA-DAG: define{{.*}} nonnull align 4 dereferenceable(4) ptr @_Z3barv(
const int &bar() {
	// CHECKA-DAG: call noundef nonnull align 4 dereferenceable(4) ptr @_ZL3foov()
	return foo();
}

#else // File B

// CHECKB-DAG: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z3barv(
const int &bar();

int main() {
	// CHECKB-DAG: call noundef nonnull align 4 dereferenceable(4) ptr @_Z3barv()
	// CHECKB-DAG: call noundef nonnull align 4 dereferenceable(4) ptr @_ZL3foov()
	&bar() == &foo() ? throw 0 : (void)0; // Should not throw exception at runtime.
}

#endif // end of Files A and B

