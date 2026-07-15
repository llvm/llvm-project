// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -Wno-vla-cxx-extension -fsycl-is-host -verify=expected %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -Wno-vla-cxx-extension -fsycl-is-device -verify=expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -Wnonportable-sycl -Wno-vla-cxx-extension -fsycl-is-host -verify=expected,nonportable %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -Wnonportable-sycl -Wno-vla-cxx-extension -fsycl-is-device -verify=expected,nonportable %s

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

// A generic kernel launch function.
template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

// Check that reference captures of kernel that defined as lambda are diagnosed.
namespace badref1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re 2{{within parameter 't' of type '(lambda at {{.*}})' declared here}}

void test() {
  int p = 0;
  double q = 0;
  float s = 0;
  // expected-note-re@+1 {{in instantiation of function template specialization 'badref1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<1>>(
      [ // expected-note{{within capture 'p' of lambda expression here}}
        // expected-note@-1{{within capture 's' of lambda expression here}}
          // expected-error@+1 {{'int &' cannot be used as the type of a kernel parameter}}
          &p, q,
          // expected-error@+1 {{'float &' cannot be used as the type of a kernel parameter}}
          &s] {
        (void)q;
        (void)p;
        (void)s;
      });
}
} // namespace badref1

// Check reference kernel parameters witin structs or lambdas;
namespace badref2 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re 3{{within parameter 't' of type '(lambda at {{.*}})' declared here}}

struct S { // expected-note 2{{within field of type 'S' declared here}}
  int a;
  int &b; //expected-error 2{{'int &' cannot be used as the type of a kernel parameter}}
};

void test() {
  int p = 0;
  auto L = [&]() { (void)p;}; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
                               // expected-note@-1 {{within capture 'p' of lambda expression here}}
  S Str {p, p};
  // expected-note-re@+1 {{in instantiation of function template specialization 'badref2::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<2>>(
      [=] { // expected-note {{within capture 'L' of lambda expression here}}
            // expected-note@-1 {{within capture 'Str' of lambda expression here}}
        (void)L;
        (void)Str;
      });

  // expected-note-re@+1 {{in instantiation of function template specialization 'badref2::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<3>>(
     [=] { // // expected-note {{within capture 'Str' of lambda expression here}}
       (void)Str;
     });

}
} // namespace badref2

// Check references within array kernel parameters.
namespace badref3 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re 3{{within parameter 't' of type '(lambda at {{.*}})' declared here}}

struct S { // expected-note {{within field of type 'S' declared here}}
  int a;
  int &b; //expected-error {{'int &' cannot be used as the type of a kernel parameter}}
};

void fooarr(int (&arr)[5]) {
}

void test(int AS) {
  int p = 0;
  S Str {p, p};
  S arr[2] = {Str, Str};
  // expected-note-re@+1 {{in instantiation of function template specialization 'badref3::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<4>>(
      [=] { // expected-note {{within capture 'arr' of lambda expression here}}
        (void)arr;
      });
  int arr1[AS];
  // expected-note-re@+1 {{in instantiation of function template specialization 'badref3::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<5>>(
      [&] { // expected-note {{within capture 'arr1' of lambda expression here}}
        (void)arr1; // expected-error {{'int (&)[AS]' cannot be used as the type of a kernel parameter}}
      });
  int arrayints[5] = {0};
  // expected-note-re@+1 {{in instantiation of function template specialization 'badref3::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<7>>(
      [&] { // expected-note {{within capture 'arrayints' of lambda expression here}}
        fooarr(arrayints); // expected-error {{'int (&)[5]' cannot be used as the type of a kernel parameter}}
      });
}
} // namespace badref3

// Check callable objects containing references.
namespace badref4 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note {{within parameter 't' of type 'badref4::Callable<int &>' declared here}}
                                // expected-note@-1 {{within parameter 't' of type 'badref4::Derived1' declared here}}
                                // expected-note@-2 {{within parameter 't' of type 'badref4::Derived2' declared here}}

template <typename T> class Callable { // expected-note 2{{within field of type 'Callable<int &>' declared here}}
  T data; // expected-error 2{{'int &' cannot be used as the type of a kernel parameter}}
public:
  Callable(T d) : data(d) {}
  void operator()() {
  }
};

class Derived1 : Callable<int> { // expected-note {{within field of type 'Derived1' declared here}}
  int &a; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
public:
  Derived1(int d, int &b) : Callable<int>(d), a(b) {}
};

class Derived2 : Callable<int&> { // expected-note {{within base class of type 'Callable<int &>' declared here}}
  int a;
public:
  Derived2(int d, int &b) : Callable<int&>(b), a(d) {}
};

void test(int AS) {
  int p = 0;
  kernel_single_task<class KN<8>>(Callable<int&>{p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'badref4::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<9>>(Callable<int>{p});
  kernel_single_task<class KN<10>>(Derived1{p, p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'badref4::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<11>>(Derived2{p, p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'badref4::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}

} // namespace badref4

// Check that a struct that hold a reference and captured by reference by lambda
// kernel object is diagnosed correctly.
namespace badref6 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} //expected-note-re {{within parameter 't' of type '(lambda at {{.*}})' declared here}}

void test() {
  int a;
  struct S {
    int &dm;
  };
  S s {a};
  kernel_single_task<class KN<13>>([&] { (void)s; });
  // expected-error@-1 {{'S &' cannot be used as the type of a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'badref6::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 's' of lambda expression here}}
}
} // namespace badref6

// Check that init capture is diagnosed correctly.
namespace badref7 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} //expected-note-re {{within parameter 't' of type '(lambda at {{.*}})' declared here}}

void test() {
  int p = 0;
  kernel_single_task<class KN<14>>([&x=p] { (void)x; });
  // expected-error@-1 {{'int &' cannot be used as the type of a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'badref7::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'x' of lambda expression here}}
}

} // namespace badref7


#include <stdatomic.h>
// Check for atomic parameters and subobjects.
namespace atomic1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // expected-note-re@-1 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // expected-note-re@-2 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // expected-note-re@-3 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // expected-note@-4 {{within parameter 't' of type 'atomic1::Kernel' declared here}}

struct Sa { 
  int a;
  _Atomic int b; // expected-error {{'_Atomic(int)' cannot be used as the type of a kernel parameter}}
                 // expected-note@-3 {{within field of type 'Sa' declared here}}
                 // expected-error@-2 {{'_Atomic(int)' cannot be used as the type of a kernel parameter}}
                 // expected-note@-5 {{within field of type 'Sa' declared here}}
                 // expected-error@-4 {{'_Atomic(int)' cannot be used as the type of a kernel parameter}}
                 // expected-note@-7 {{within field of type 'Sa' declared here}}
};

class Kernel {
  Sa data{1, 2}; // expected-note@-1 {{within field of type 'Kernel' declared here}}
public:
  void operator()() { }
};

void test() {
  _Atomic int a = 0;
  _Atomic(int) b = 2;
  Sa s{1, 2};
  Sa arr[] = {s, s};
  kernel_single_task<class KN<15>>([=]{ (void)a; });
  // expected-error@-1 {{'_Atomic(int)' cannot be used as the type of a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'atomic1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'a' of lambda expression here}}
  kernel_single_task<class KN<16>>([=]{ (void)b; });
  // expected-error@-1 {{'_Atomic(int)' cannot be used as the type of a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'atomic1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'b' of lambda expression here}}
  kernel_single_task<class KN<17>>([=]{ (void)s; });
  // expected-note-re@-1 {{in instantiation of function template specialization 'atomic1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-2 {{within capture 's' of lambda expression here}}
  kernel_single_task<class KN<18>>([=]{ (void)arr; });
  // expected-note-re@-1 {{in instantiation of function template specialization 'atomic1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-2 {{within capture 'arr' of lambda expression here}}
  kernel_single_task<class KN<19>>(Kernel{});
  // expected-note-re@-1 {{in instantiation of function template specialization 'atomic1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}

} // namespace atomic1

// Check for flexible array members -- would not be copyable to device
namespace fam1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note {{within parameter 't' of type 'fam1::Kernel' declared here}}

struct FAM { 
  int a;
  int b[];
};

class Kernel { // expected-note {{within field of type 'Kernel' declared here}}
  FAM fam; // expected-error {{'FAM' contains a flexible array member and cannot be used as a SYCL kernel parameter}}
public:
  void operator()() { (void)fam; }
};

void test() {
  FAM fam;
  // Flexible array members cannot be captured in a lambda, thus no lambda tests are provided.
  kernel_single_task<class KN<21>>(Kernel{});
  // expected-note-re@-1 {{in instantiation of function template specialization 'fam1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}

} // namespace fam1

// Check for pointer parameters
namespace nonportable1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // nonportable-note-re {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // nonportable-note-re@-1 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // nonportable-note-re@-2 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}
                                // nonportable-note-re@-3 {{within parameter 't' of type '(lambda at {{.*}})' declared here}}

struct S { // nonportable-note {{within field of type 'S' declared here}}
           // nonportable-note@-1 {{within field of type 'S' declared here}}
  int *ptr;
  // nonportable-warning@-1 {{pointers used in SYCL kernels must point to device-accessible memory, i.e. the USM}}
  // nonportable-warning@-2 {{pointers used in SYCL kernels must point to device-accessible memory, i.e. the USM}}
};

class C { // nonportable-note {{within field of type 'C' declared here}}
private:
  int *ptr;
  // nonportable-warning@-1 {{pointers used in SYCL kernels must point to device-accessible memory, i.e. the USM}}
public:
  C(int *p) : ptr(p) {}
};

void test() {
  int *ptr;
  kernel_single_task<class KN<25>>([=]{ (void)ptr; });
  // nonportable-warning@-1 {{pointers used in SYCL kernels must point to device-accessible memory, i.e. the USM}}
  // nonportable-note-re@-2 {{in instantiation of function template specialization 'nonportable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // nonportable-note@-3 {{within capture 'ptr' of lambda expression here}}

  S s{ptr};
  kernel_single_task<class KN<26>>([=]{ (void)s; });
  // nonportable-note-re@-1 {{in instantiation of function template specialization 'nonportable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // nonportable-note@-2 {{within capture 's' of lambda expression here}}
  
  C c{ptr};
  kernel_single_task<class KN<27>>([=]{ (void)c; });
  // nonportable-note-re@-1 {{in instantiation of function template specialization 'nonportable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // nonportable-note@-2 {{within capture 'c' of lambda expression here}}
  
  S arr[3] = {{ptr}, {ptr}, {ptr}};
  kernel_single_task<class KN<28>>([=]{ (void)arr; });
  // nonportable-note-re@-1 {{in instantiation of function template specialization 'nonportable1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // nonportable-note@-2 {{within capture 'arr' of lambda expression here}}
}

} // namespace nonportable1

// Check for virtual bases
namespace vbase1 {
// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T t) {} // expected-note-re {{within parameter 't' of type '(lambda at {{.*}})' declared here}}

class Base { 
  // No diagnostic is issued for data because recursive subobject visitation
  // stops once a virtual base class is found.
  int &data; 
public:
  Base(int &a) : data(a) {}
};

class Derived : virtual Base { 
public:
  Derived(int &a) : Base(a) {}

};

void test() {
  int p = 0;
  Derived d{p};
  kernel_single_task<class KN<29>>([=]{ (void)d; });
  // expected-error@-1 {{'Derived' inherits virtual base classes and cannot be used as a SYCL kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'vbase1::kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note@-3 {{within capture 'd' of lambda expression here}}
}
} // namespace vbase1
