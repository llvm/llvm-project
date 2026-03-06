// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsycl-is-host -verify=host %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsycl-is-device -verify=device %s

// These tests validate that a diagnostic is issued if a function declared with
// the sycl_kernel_entry_point attribute is ODR-used from code that is emitted
// during device compilation. Such uses are ill-formed because such functions
// are used to define an offload kernel entry point function; they aren't
// available for ordinary function use.

// host-no-diagnostics

// Emulate inclusion of <typeinfo>.
namespace std {
struct type_info {
  virtual ~type_info();
};
} // namespace std

// A generic kernel launch function.
template<typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

// A kernel name type template.
template<int> struct KN;

// SYCL kernel entry point functions. These are used to both trigger the
// emission of a function during device compilation (but not during host
// compilation) and to trigger a diagnostic if ODR-used from a function
// emitted during device compilation.
// device-note@+1 4 {{attribute is here}}
[[clang::sycl_kernel_entry_point(KN<1>)]]
void skep();
struct SKL {
  // device-note@+1 6 {{attribute is here}}
  [[clang::sycl_kernel_entry_point(KN<2>)]]
  void mskep();
  // device-note@+1 6 {{attribute is here}}
  [[clang::sycl_kernel_entry_point(KN<3>)]]
  static void smskep();
  // device-note@+1 2 {{attribute is here}}
  [[clang::sycl_kernel_entry_point(KN<4>)]]
  void operator()() const;
};

// A function that is emitted on the device due to usage reachable from a
// SYCL kernel entry point function. ODR-uses of sycl_kernel_entry_point
// attributed functions within this function require a diagnostic during
// device compilation.
void df() {
  // Not ODR-uses; ok.
  decltype(&skep) p1 = nullptr;
  decltype(&SKL::mskep) p2 = nullptr;
  decltype(&SKL::smskep) p3 = nullptr;

  // Not ODR-uses; ok.
  (void)noexcept(skep());
  (void)noexcept(SKL{}.mskep());
  (void)noexcept(SKL::smskep());

  // Not ODR-uses; ok.
  (void)typeid(&skep);
  (void)typeid(&SKL::mskep);
  (void)typeid(&SKL::smskep);

  // device-error@+1 2 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 2 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();

  // device-error@+1 2 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  (void)&skep;
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  (void)&SKL::mskep;
  // device-error@+1 2 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  (void)&SKL::smskep;

  SKL sklo;
  // device-error@+1 2 {{function 'operator()' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  sklo();
}

// device-note@+1 5 {{attribute is here}}
[[clang::sycl_kernel_entry_point(KN<1>)]]
void skep() {
  // device-note@+1 {{called by 'skep'}}
  df();
  // device-error@+1 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();
}

// device-note@+1 7 {{attribute is here}}
[[clang::sycl_kernel_entry_point(KN<2>)]]
void SKL::mskep() {
  df();
  // device-error@+1 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();
}

// device-note@+1 3 {{attribute is here}}
[[clang::sycl_kernel_entry_point(KN<3>)]]
void SKL::smskep() {
  df();
  // device-error@+1 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();
}

[[clang::sycl_kernel_entry_point(KN<4>)]]
void SKL::operator()() const {
  df();
  // device-error@+1 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 2 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();
}

[[clang::sycl_external]]
void sedf() {
  // device-note@+1 {{called by 'sedf'}}
  df();
  // device-error@+1 {{function 'skep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  skep();
  // device-error@+1 {{function 'mskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL{}.mskep();
  // device-error@+1 {{function 'smskep' cannot be used in device code because it is declared with the 'clang::sycl_kernel_entry_point' attribute}}
  SKL::smskep();
}
