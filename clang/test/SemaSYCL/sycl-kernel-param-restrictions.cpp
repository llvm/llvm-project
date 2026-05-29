// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -Wno-vla-cxx-extension -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple spirv64 -std=c++17 -fsyntax-only -Wno-vla-cxx-extension -fsycl-is-device -verify %s

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

// A generic kernel launch function.
template<typename KNT, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

// Kernel entry point template definition.
template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void kernel_single_task(T) {}

// Check that reference captures of kernel that defined as lambda are diagnosed.
namespace badref1 {
void test() {
  int p = 0;
  double q = 0;
  float s = 0;
  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<1>>(
      [ // expected-note2{{within field of type}}
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
struct S { // expected-note 2{{within field of type 'S' declared here}}
  int a;
  int &b; //expected-error 2{{'int &' cannot be used as the type of a kernel parameter}}
};

void test() {
  int p = 0;
  auto L = [&]() { (void)p;}; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
                               // expected-note@-1 {{within field of type}}
  S Str {p, p};
  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<2>>(
      [=] { // expected-note 2{{within field of type}}
        (void)L;
        (void)Str;
      });

  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<3>>(
     [=] { // // expected-note {{within field of type}}
       (void)Str;
     });

}
} // namespace badref2

// Check references within array kernel parameters.
namespace badref3 {
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
  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<4>>(
      [=] { // expected-note {{within field of type}}
        (void)arr;
      });
  int arr1[AS];
  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<5>>(
      [&] { // expected-note {{within field of type}}
        (void)arr1; // expected-error {{'int (&)[AS]' cannot be used as the type of a kernel parameter}}
      });
  int arrayints[5] = {0};
  // expected-note-re@+1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<7>>(
      [&] { // expected-note {{within field of type}}
        fooarr(arrayints); // expected-error {{'int (&)[5]' cannot be used as the type of a kernel parameter}}
      });
}
} // namespace badref3

// Check callable objects containing references.
namespace badref4 {
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

class Derived2 : Callable<int&> { // expected-note {{within base of type 'Callable<int &>' declared here}}
  int a;
public:
  Derived2(int d, int &b) : Callable<int&>(b), a(d) {}
};

void test(int AS) {
  int p = 0;
  kernel_single_task<class KN<8>>(Callable<int&>{p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<9>>(Callable<int>{p});
  kernel_single_task<class KN<10>>(Derived1{p, p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  kernel_single_task<class KN<11>>(Derived2{p, p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}

} // namespace badref4

// Test virtual bases.
// FIXME: explicitly diagnose virtual bases within kernel parameters.
namespace badref5 {

class Base { // expected-note {{within field of type 'Base' declared here}}}
  int &data; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
public:
  Base(int &a) : data(a) {}
};

class Derived : virtual Base { // expected-note {{within base of type 'Base' declared here}}
public:
  Derived(int &a) : Base(a) {}

};

void test() {
  int p = 0;
  kernel_single_task<class KN<12>>(Derived{p});
  // expected-note-re@-1 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
}
} // namespace badref5

// Check that a struct that hold a reference and captured by reference by lambda
// kernel object is diagnosed correctly.
namespace badref6 {
void test() {
  int a;
  struct S {
    int &dm;
  };
  S s {a};
  kernel_single_task<class KN<13>>([&] { (void)s; });
  // expected-error@-1 {{'S &' cannot be used as the type of a kernel parameter}}
  // expected-note-re@-2 {{in instantiation of function template specialization 'kernel_single_task<KN<{{[0-9]+}}>, {{.*}}>' requested here}}
  // expected-note-re@-3 {{within field of type '(lambda at {{.*}} declared here}}
}

} // namespace badref7
