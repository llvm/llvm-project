// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++1y -O1 -disable-llvm-passes %s -o - | FileCheck %s -check-prefix=CHECKA -check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++1y -O1 -disable-llvm-passes -fcxx-exceptions %s -o - | FileCheck %s -check-prefix=CHECKB -check-prefix=CHECK
// expected-no-diagnostics

// The variable template specialization x<Foo> generated in each file
// should be 'internal global' and not 'linkonce_odr global'.

template <typename T> int x = 42;

// CHECK-DAG: @_Z1xIZL3foovE3FooE = internal global

// CHECK-DAG: define internal dereferenceable(4) i32* @_ZL3foov(
static int &foo() {
   struct Foo { };
   
   // CHECK-DAG: ret i32* @_Z1xIZL3foovE3FooE
   return x<Foo>;
}


#if !__has_feature(cxx_exceptions) // File A
// CHECKA-DAG: define dereferenceable(4) i32* @_Z3barv(
int &bar() { 
	// CHECKA-DAG: call dereferenceable(4) i32* @_ZL3foov()
	return foo();
}

#else // File B

// CHECKB-DAG: declare dereferenceable(4) i32* @_Z3barv(
int &bar();

int main() {
	// CHECKB-DAG: call dereferenceable(4) i32* @_Z3barv()
	// CHECKB-DAG: call dereferenceable(4) i32* @_ZL3foov()
	&bar() == &foo() ? throw 0 : (void)0; // Should not throw exception at runtime.
}

#endif // end of Files A and B

