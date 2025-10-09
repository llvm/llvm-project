// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++17 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-device -fcxx-exceptions -verify=device,expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-host -fcxx-exceptions -verify=host,expected %s

// A unique kernel name type is required for each declared kernel entry point.
template<int, int = 0> struct KN;

[[clang::sycl_kernel_entry_point(KN<1>)]]
void nolauncher() {} 
// host-error@-1 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
// device-warning@-2 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
// expected-note@-3 {{define 'sycl_kernel_launch' function template to fix}}

void sycl_kernel_launch(const char *, int arg);
// expected-note@-1 {{declared as a non-template here}}

[[clang::sycl_kernel_entry_point(KN<2>)]]
void nontemplatel() {}
// host-error@-1 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
// device-warning@-2 {{unable to find suitable 'sycl_kernel_launch' function for host code synthesis}}
// expected-note@-3 {{define 'sycl_kernel_launch' function template to fix}}
// expected-error@-4 {{'sycl_kernel_launch' following the 'template' keyword does not refer to a template}}

template <typename KernName>
void sycl_kernel_launch(const char *, int arg);
// expected-note@-1 {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
// expected-note@-2 2{{candidate function template not viable: no known conversion from 'Kern' to 'int' for 2nd argument}}

[[clang::sycl_kernel_entry_point(KN<3>)]]
void notenoughargs() {}
// expected-error@-1 {{no matching function for call to 'sycl_kernel_launch'}}
// FIXME: Should this also say "no suitable function for host code synthesis"?


template <typename KernName>
void sycl_kernel_launch(const char *, bool arg = 1);
// expected-note@-1 2{{candidate function template not viable: no known conversion from 'Kern' to 'bool' for 2nd argument}}

[[clang::sycl_kernel_entry_point(KN<4>)]]
void enoughargs() {}

namespace boop {
template <typename KernName, typename KernelObj>
void sycl_kernel_launch(const char *, KernelObj);

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
void iboop(KernelObj Kernel) {
  Kernel();
}
}

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
void idontboop(KernelObj Kernel) {
  Kernel();
}
// expected-error@-3 {{no matching function for call to 'sycl_kernel_launch'}}

struct Kern {
  int a;
  int *b;
  Kern(int _a, int* _b) : a(_a), b(_b) {}
  void operator()(){ *b = a;}
};

void foo() {
  int *a;
  Kern b(1, a);
  idontboop<KN<6>>(b);
  // expected-note@-1 {{in instantiation of function template specialization 'idontboop<KN<6>, Kern>' requested here}}
  boop::iboop<KN<7>>(b);
}

class MaybeHandler {

template <typename KernName>
void sycl_kernel_launch(const char *);

template <typename KernName, typename... Tys>
void sycl_kernel_launch(const char *, Tys ...Args);

public:

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
void entry(KernelObj Kernel) {
  Kernel();
}
};

class MaybeHandler2 {

template <typename KernName, typename... Tys>
static void sycl_kernel_launch(const char *, Tys ...Args);

public:

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
void entry(KernelObj Kernel) {
  Kernel();
}
};

class MaybeHandler3 {

template <typename KernName, typename... Tys>
static void sycl_kernel_launch(const char *, Tys ...Args);

public:

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
static void entry(KernelObj Kernel) {
  Kernel();
}
};

class MaybeHandler4 {

template <typename KernName, typename... Tys>
void sycl_kernel_launch(const char *, Tys ...Args);

public:

template <typename KernName, typename KernelObj>
[[clang::sycl_kernel_entry_point(KernName)]]
static void entry(KernelObj Kernel) { 
  // expected-error@-1 {{call to non-static member function without an object argument}}
  // FIXME: Should that be clearer?
  Kernel();
}
};

template<typename>
struct base_handler {
  template<typename KNT, typename... Ts>
  void sycl_kernel_launch(const char*, Ts...) {}
};
struct derived_handler : base_handler<derived_handler> {
  template<typename KNT, typename KT>
  [[clang::sycl_kernel_entry_point(KNT)]]
  void entry(KT k) { k(); }
};

template<int N>
struct derived_handler_t : base_handler<derived_handler_t<N>> {
  template<typename KNT, typename KT>
// FIXME this fails because accessing members of dependent bases requires
// explicit qualification.
  [[clang::sycl_kernel_entry_point(KNT)]]
  void entry(KT k) { k(); }
  // expected-error@-1 {{no matching function for call to 'sycl_kernel_launch'}}
};

template<typename KNT>
struct kernel_launcher {
  template<typename... Ts>
  void operator()(const char*, Ts...) const {}
};

namespace var {
template<typename KNT>
kernel_launcher<KNT> sycl_kernel_launch;

struct handler {
  template<typename KNT, typename KT>
  [[clang::sycl_kernel_entry_point(KNT)]]
  void entry(KT k) { k(); }
};
}


void bar() {
  int *a;
  Kern b(1, a);
  MaybeHandler H;
  MaybeHandler2 H1;
  MaybeHandler3 H2;
  MaybeHandler4 H3;
  H.entry<KN<8>>(b);
  H1.entry<KN<9>>(b);
  H2.entry<KN<10>>(b);
  H3.entry<KN<11>>(b);

  derived_handler H5;
  H5.entry<KN<12>>(b);

  derived_handler_t<13> H6;
  H6.entry<KN<13>>(b); //expected-note {{in instantiation of function template specialization}}

  var::handler h;
  h.entry<KN<14>>(b);
}



