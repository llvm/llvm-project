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

struct Empty {};

struct S {
  int a;
  int &b; //expected-error 2{{'int &' cannot be used as the type of a kernel parameter}}
};

void fooarr(int (&arr)[5]) {
}

template <typename T> class Callable {
  T data; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
public:
  Callable(T d) : data(d) {}
  void operator()() {
  }
};

void refCases(int AS) {
  int p = 0;
  double q = 0;
  float s = 0;
  kernel_single_task<class KN<1>>( // expected-note {{requested here}}
      [
          // expected-error@+1 {{'int &' cannot be used as the type of a kernel parameter}}
          &p, q,
          // expected-error@+1 {{'float &' cannot be used as the type of a kernel parameter}}
          &s] {
        (void)q;
        (void)p;
        (void)s;
      });

   auto L = [&]() { (void)p;}; // expected-error {{'int &' cannot be used as the type of a kernel parameter}}
   S Str {p, p};
  kernel_single_task<class KN<2>>( // expected-note {{requested here}}
      [=] {
        (void)L;
        (void)Str; // no error because fail for L already
      });

  kernel_single_task<class KN<3>>( // expected-note {{requested here}}
     [=] {
       (void)Str;
     });

  S arr[2] = {Str, Str};
  kernel_single_task<class KN<4>>( // expected-note {{requested here}}
      [=] {
        (void)arr;
      });
  int arr1[AS];
  kernel_single_task<class KN<5>>( // expected-note {{requested here}}
      [&] {
        (void)arr1; // expected-error {{'int (&)[AS]' cannot be used as the type of a kernel parameter}}
      });
  auto a = &arr1;
  kernel_single_task<class KN<6>>( // expected-note {{requested here}}
      [=] {
        (void)a; // expected-error {{'int[AS]' cannot be used as the type of a kernel parameter}}
      });
  int arrayints[5] = {0};
  kernel_single_task<class KN<7>>( // expected-note {{requested here}}
      [&] {
        fooarr(arrayints); // expected-error {{'int (&)[5]' cannot be used as the type of a kernel parameter}}
      });
  kernel_single_task<class KN<8>>(Callable<int&>{p}); // expected-note {{requested here}}
  kernel_single_task<class KN<9>>(Callable<int>{p});
}

