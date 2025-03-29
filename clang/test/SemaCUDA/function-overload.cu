// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -verify=host,hostdefer,devdefer,expected %s
// RUN: %clang_cc1 -std=c++14 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:   -fcuda-is-device -verify=dev,devnodeferonly,hostdefer,devdefer,expected %s
// RUN: %clang_cc1 -fgpu-exclude-wrong-side-overloads -fgpu-defer-diag -DDEFER=1 \
// RUN:    -std=c++14 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:    -verify=host,hostdefer,expected %s
// RUN: %clang_cc1 -fgpu-exclude-wrong-side-overloads -fgpu-defer-diag -DDEFER=1 \
// RUN:    -std=c++14 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device \
// RUN:    -verify=dev,devdeferonly,devdefer,expected %s

#include "Inputs/cuda.h"

// Opaque return types used to check that we pick the right overloads.
struct HostReturnTy {};
struct HostReturnTy2 {};
struct DeviceReturnTy {};
struct DeviceReturnTy2 {};
struct HostDeviceReturnTy {};
struct TemplateReturnTy {};

typedef HostReturnTy (*HostFnPtr)();
typedef DeviceReturnTy (*DeviceFnPtr)();
typedef HostDeviceReturnTy (*HostDeviceFnPtr)();
typedef void (*GlobalFnPtr)();  // __global__ functions must return void.

// CurrentReturnTy is {HostReturnTy,DeviceReturnTy} during {host,device}
// compilation.
#ifdef __CUDA_ARCH__
typedef DeviceReturnTy CurrentReturnTy;
#else
typedef HostReturnTy CurrentReturnTy;
#endif

// CurrentFnPtr is a function pointer to a {host,device} function during
// {host,device} compilation.
typedef CurrentReturnTy (*CurrentFnPtr)();

// Host and unattributed functions can't be overloaded.
__host__ void hh() {} // expected-note {{previous definition is here}}
void hh() {} // expected-error {{redefinition of 'hh'}}

// H/D overloading is OK.
__host__ HostReturnTy dh() { return HostReturnTy(); }
__device__ DeviceReturnTy dh() { return DeviceReturnTy(); }

// H/HD and D/HD are not allowed.
__host__ __device__ int hdh() { return 0; } // expected-note {{previous declaration is here}}
__host__ int hdh() { return 0; }
// expected-error@-1 {{__host__ function 'hdh' cannot overload __host__ __device__ function 'hdh'}}

__host__ int hhd() { return 0; }            // expected-note {{previous declaration is here}}
__host__ __device__ int hhd() { return 0; }
// expected-error@-1 {{__host__ __device__ function 'hhd' cannot overload __host__ function 'hhd'}}

__host__ __device__ int hdd() { return 0; } // expected-note {{previous declaration is here}}
__device__ int hdd() { return 0; }
// expected-error@-1 {{__device__ function 'hdd' cannot overload __host__ __device__ function 'hdd'}}

__device__ int dhd() { return 0; }          // expected-note {{previous declaration is here}}
__host__ __device__ int dhd() { return 0; }
// expected-error@-1 {{__host__ __device__ function 'dhd' cannot overload __device__ function 'dhd'}}

// Same tests for extern "C" functions.
extern "C" __host__ int chh() { return 0; } // expected-note {{previous definition is here}}
extern "C" int chh() { return 0; }          // expected-error {{redefinition of 'chh'}}

// H/D overloading is OK.
extern "C" __device__ DeviceReturnTy cdh() { return DeviceReturnTy(); }
extern "C" __host__ HostReturnTy cdh() { return HostReturnTy(); }

// H/HD and D/HD overloading is not allowed.
extern "C" __host__ __device__ int chhd1() { return 0; } // expected-note {{previous declaration is here}}
extern "C" __host__ int chhd1() { return 0; }
// expected-error@-1 {{__host__ function 'chhd1' cannot overload __host__ __device__ function 'chhd1'}}

extern "C" __host__ int chhd2() { return 0; } // expected-note {{previous declaration is here}}
extern "C" __host__ __device__ int chhd2() { return 0; }
// expected-error@-1 {{__host__ __device__ function 'chhd2' cannot overload __host__ function 'chhd2'}}

// Helper functions to verify calling restrictions.
__device__ DeviceReturnTy d() { return DeviceReturnTy(); }
// host-note@-1 1+ {{'d' declared here}}
// hostdefer-note@-2 1+ {{candidate function not viable: call to __device__ function from __host__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __device__ function from __host__ __device__ function}}

__host__ HostReturnTy h() { return HostReturnTy(); }
// dev-note@-1 1+ {{'h' declared here}}
// devdefer-note@-2 1+ {{candidate function not viable: call to __host__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
// devdefer-note@-4 1+ {{candidate function not viable: call to __host__ function from __global__ function}}

__global__ void g() {}
// dev-note@-1 1+ {{'g' declared here}}
// devdefer-note@-2 1+ {{candidate function not viable: call to __global__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __global__ function from __host__ __device__ function}}
// devdefer-note@-4 1+ {{candidate function not viable: call to __global__ function from __global__ function}}

extern "C" __device__ DeviceReturnTy cd() { return DeviceReturnTy(); }
// host-note@-1 1+ {{'cd' declared here}}
// hostdefer-note@-2 1+ {{candidate function not viable: call to __device__ function from __host__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __device__ function from __host__ __device__ function}}

extern "C" __host__ HostReturnTy ch() { return HostReturnTy(); }
// dev-note@-1 1+ {{'ch' declared here}}
// devdefer-note@-2 1+ {{candidate function not viable: call to __host__ function from __device__ function}}
// expected-note@-3 0+ {{candidate function not viable: call to __host__ function from __host__ __device__ function}}
// devdefer-note@-4 1+ {{candidate function not viable: call to __host__ function from __global__ function}}

__host__ void hostf() {
  DeviceFnPtr fp_d = d;         // host-error {{reference to __device__ function 'd' in __host__ function}}
  DeviceReturnTy ret_d = d();   // hostdefer-error {{no matching function for call to 'd'}}
  DeviceFnPtr fp_cd = cd;       // host-error {{reference to __device__ function 'cd' in __host__ function}}
  DeviceReturnTy ret_cd = cd(); // hostdefer-error {{no matching function for call to 'cd'}}

  HostFnPtr fp_h = h;
  HostReturnTy ret_h = h();
  HostFnPtr fp_ch = ch;
  HostReturnTy ret_ch = ch();

  HostFnPtr fp_dh = dh;
  HostReturnTy ret_dh = dh();
  HostFnPtr fp_cdh = cdh;
  HostReturnTy ret_cdh = cdh();

  GlobalFnPtr fp_g = g;
  g(); // expected-error {{call to global function 'g' not configured}}
  g<<<0, 0>>>();
}

__device__ void devicef() {
  DeviceFnPtr fp_d = d;
  DeviceReturnTy ret_d = d();
  DeviceFnPtr fp_cd = cd;
  DeviceReturnTy ret_cd = cd();

  HostFnPtr fp_h = h;         // dev-error {{reference to __host__ function 'h' in __device__ function}}
  HostReturnTy ret_h = h();   // devdefer-error {{no matching function for call to 'h'}}
  HostFnPtr fp_ch = ch;       // dev-error {{reference to __host__ function 'ch' in __device__ function}}
  HostReturnTy ret_ch = ch(); // devdefer-error {{no matching function for call to 'ch'}}

  DeviceFnPtr fp_dh = dh;
  DeviceReturnTy ret_dh = dh();
  DeviceFnPtr fp_cdh = cdh;
  DeviceReturnTy ret_cdh = cdh();

  GlobalFnPtr fp_g = g; // dev-error {{reference to __global__ function 'g' in __device__ function}}
  g(); // devdefer-error {{no matching function for call to 'g'}}
  g<<<0,0>>>(); // dev-error {{reference to __global__ function 'g' in __device__ function}}
}

__global__ void globalf() {
  DeviceFnPtr fp_d = d;
  DeviceReturnTy ret_d = d();
  DeviceFnPtr fp_cd = cd;
  DeviceReturnTy ret_cd = cd();

  HostFnPtr fp_h = h;         // dev-error {{reference to __host__ function 'h' in __global__ function}}
  HostReturnTy ret_h = h();   // devdefer-error {{no matching function for call to 'h'}}
  HostFnPtr fp_ch = ch;       // dev-error {{reference to __host__ function 'ch' in __global__ function}}
  HostReturnTy ret_ch = ch(); // devdefer-error {{no matching function for call to 'ch'}}

  DeviceFnPtr fp_dh = dh;
  DeviceReturnTy ret_dh = dh();
  DeviceFnPtr fp_cdh = cdh;
  DeviceReturnTy ret_cdh = cdh();

  GlobalFnPtr fp_g = g; // dev-error {{reference to __global__ function 'g' in __global__ function}}
  g(); // devdefer-error {{no matching function for call to 'g'}}
  g<<<0,0>>>(); // dev-error {{reference to __global__ function 'g' in __global__ function}}
}

__host__ __device__ void hostdevicef() {
  DeviceFnPtr fp_d = d;
  DeviceReturnTy ret_d = d();
  DeviceFnPtr fp_cd = cd;
  DeviceReturnTy ret_cd = cd();
#if !defined(__CUDA_ARCH__)
  // expected-error@-5 {{reference to __device__ function 'd' in __host__ __device__ function}}
  // expected-error@-5 {{reference to __device__ function 'd' in __host__ __device__ function}}
  // expected-error@-5 {{reference to __device__ function 'cd' in __host__ __device__ function}}
  // expected-error@-5 {{reference to __device__ function 'cd' in __host__ __device__ function}}
#endif

  HostFnPtr fp_h = h;
  HostReturnTy ret_h = h();
  HostFnPtr fp_ch = ch;
  HostReturnTy ret_ch = ch();
#if defined(__CUDA_ARCH__)
  // expected-error@-5 {{reference to __host__ function 'h' in __host__ __device__ function}}
  // expected-error@-5 {{reference to __host__ function 'h' in __host__ __device__ function}}
  // devdefer-error@-5 {{reference to __host__ function 'ch' in __host__ __device__ function}}
  // expected-error@-5 {{reference to __host__ function 'ch' in __host__ __device__ function}}
#endif

  CurrentFnPtr fp_dh = dh;
  CurrentReturnTy ret_dh = dh();
  CurrentFnPtr fp_cdh = cdh;
  CurrentReturnTy ret_cdh = cdh();

  GlobalFnPtr fp_g = g;
#if defined(__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif

  g();
#if defined (__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#else
  // expected-error@-4 {{call to global function 'g' not configured}}
#endif

  g<<<0,0>>>();
#if defined(__CUDA_ARCH__)
  // expected-error@-2 {{reference to __global__ function 'g' in __host__ __device__ function}}
#endif
}

// Test for address of overloaded function resolution in the global context.
HostFnPtr fp_h = h;
HostFnPtr fp_ch = ch;
#if defined (__CUDA_ARCH__)
__device__
#endif
CurrentFnPtr fp_dh = dh;
#if defined (__CUDA_ARCH__)
__device__
#endif
CurrentFnPtr fp_cdh = cdh;
GlobalFnPtr fp_g = g;


// Test overloading of destructors
// Can't mix H and unattributed destructors
struct d_h {
  ~d_h() {} // expected-note {{previous definition is here}}
  __host__ ~d_h() {} // expected-error {{destructor cannot be redeclared}}
};

// HD is OK
struct d_hd {
  __host__ __device__ ~d_hd() {}
};

// Test overloading of member functions
struct m_h {
  void operator delete(void *ptr); // expected-note {{previous declaration is here}}
  __host__ void operator delete(void *ptr); // expected-error {{class member cannot be redeclared}}
};

// D/H overloading is OK
struct m_dh {
  __device__ void operator delete(void *ptr);
  __host__ void operator delete(void *ptr);
};

// HD by itself is OK
struct m_hd {
  __device__ __host__ void operator delete(void *ptr);
};

struct m_hhd {
  __host__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ __device__ void operator delete(void *ptr) {}
  // expected-error@-1 {{__host__ __device__ function 'operator delete' cannot overload __host__ function 'operator delete'}}
};

struct m_hdh {
  __host__ __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ void operator delete(void *ptr) {}
  // expected-error@-1 {{__host__ function 'operator delete' cannot overload __host__ __device__ function 'operator delete'}}
};

struct m_dhd {
  __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __host__ __device__ void operator delete(void *ptr) {}
  // expected-error@-1 {{__host__ __device__ function 'operator delete' cannot overload __device__ function 'operator delete'}}
};

struct m_hdd {
  __host__ __device__ void operator delete(void *ptr) {} // expected-note {{previous declaration is here}}
  __device__ void operator delete(void *ptr) {}
  // expected-error@-1 {{__device__ function 'operator delete' cannot overload __host__ __device__ function 'operator delete'}}
};

// __global__ functions can't be overloaded based on attribute
// difference.
struct G {
  friend void friend_of_g(G &arg); // expected-note {{previous declaration is here}}
private:
  int x; // expected-note {{declared private here}}
};
__global__ void friend_of_g(G &arg) { int x = arg.x; }
// expected-error@-1 {{__global__ function 'friend_of_g' cannot overload __host__ function 'friend_of_g'}}
// expected-error@-2 {{'x' is a private member of 'G'}}
void friend_of_g(G &arg) { int x = arg.x; }

// HD functions are sometimes allowed to call H or D functions -- this
// is an artifact of the source-to-source splitting performed by nvcc
// that we need to mimic. During device mode compilation in nvcc, host
// functions aren't present at all, so don't participate in
// overloading. But in clang, H and D functions are present in both
// compilation modes. Clang normally uses the target attribute as a
// tiebreaker between overloads with otherwise identical priority, but
// in order to match nvcc's behavior, we sometimes need to wholly
// discard overloads that would not be present during compilation
// under nvcc.

template <typename T> TemplateReturnTy template_vs_function(T arg) {
  return TemplateReturnTy();
}
__device__ DeviceReturnTy template_vs_function(float arg) {
  return DeviceReturnTy();
}

// Here we expect to call the templated function during host compilation, even
// if -fcuda-disable-target-call-checks is passed, and even though C++ overload
// rules prefer the non-templated function.
__host__ __device__ void test_host_device_calls_template(void) {
#ifdef __CUDA_ARCH__
  typedef DeviceReturnTy ExpectedReturnTy;
#else
  typedef TemplateReturnTy ExpectedReturnTy;
#endif

  ExpectedReturnTy ret1 = template_vs_function(1.0f);
  ExpectedReturnTy ret2 = template_vs_function(2.0);
}

// Calls from __host__ and __device__ functions should always call the
// overloaded function that matches their mode.
__host__ void test_host_calls_template_fn() {
  TemplateReturnTy ret1 = template_vs_function(1.0f);
  TemplateReturnTy ret2 = template_vs_function(2.0);
}

__device__ void test_device_calls_template_fn() {
  DeviceReturnTy ret1 = template_vs_function(1.0f);
  DeviceReturnTy ret2 = template_vs_function(2.0);
}

// If we have a mix of HD and H-only or D-only candidates in the overload set,
// normal C++ overload resolution rules apply first.
template <typename T> TemplateReturnTy template_vs_hd_function(T arg)
// devnodeferonly-note@-1{{'template_vs_hd_function<int>' declared here}}
{
  return TemplateReturnTy();
}
__host__ __device__ HostDeviceReturnTy template_vs_hd_function(float arg) {
  return HostDeviceReturnTy();
}

__host__ __device__ void test_host_device_calls_hd_template() {
#if __CUDA_ARCH__ && DEFER
  typedef HostDeviceReturnTy ExpectedReturnTy;
#else
  typedef TemplateReturnTy ExpectedReturnTy;
#endif
  HostDeviceReturnTy ret1 = template_vs_hd_function(1.0f);
  ExpectedReturnTy ret2 = template_vs_hd_function(1);
  // devnodeferonly-error@-1{{reference to __host__ function 'template_vs_hd_function<int>' in __host__ __device__ function}}
}

__host__ void test_host_calls_hd_template() {
  HostDeviceReturnTy ret1 = template_vs_hd_function(1.0f);
  TemplateReturnTy ret2 = template_vs_hd_function(1);
}

__device__ void test_device_calls_hd_template() {
  HostDeviceReturnTy ret1 = template_vs_hd_function(1.0f);
  // Host-only function template is not callable with strict call checks,
  // so for device side HD function will be the only choice.
  HostDeviceReturnTy ret2 = template_vs_hd_function(1);
}

// Check that overloads still work the same way on both host and
// device side when the overload set contains only functions from one
// side of compilation.
__device__ DeviceReturnTy device_only_function(int arg) { return DeviceReturnTy(); }
__device__ DeviceReturnTy2 device_only_function(float arg) { return DeviceReturnTy2(); }
#ifndef __CUDA_ARCH__
  // expected-note@-3 2{{'device_only_function' declared here}}
  // expected-note@-3 2{{'device_only_function' declared here}}
#endif
__host__ HostReturnTy host_only_function(int arg) { return HostReturnTy(); }
__host__ HostReturnTy2 host_only_function(float arg) { return HostReturnTy2(); }
#ifdef __CUDA_ARCH__
  // expected-note@-3 2{{'host_only_function' declared here}}
  // expected-note@-3 2{{'host_only_function' declared here}}
#endif

__host__ __device__ void test_host_device_single_side_overloading() {
  DeviceReturnTy ret1 = device_only_function(1);
  DeviceReturnTy2 ret2 = device_only_function(1.0f);
#ifndef __CUDA_ARCH__
  // expected-error@-3 {{reference to __device__ function 'device_only_function' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __device__ function 'device_only_function' in __host__ __device__ function}}
#endif
  HostReturnTy ret3 = host_only_function(1);
  HostReturnTy2 ret4 = host_only_function(1.0f);
#ifdef __CUDA_ARCH__
  // expected-error@-3 {{reference to __host__ function 'host_only_function' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __host__ function 'host_only_function' in __host__ __device__ function}}
#endif
}

// wrong-sided overloading should not cause diagnostic unless it is emitted.
// This inline function is not emitted.
inline __host__ __device__ void test_host_device_wrong_side_overloading_inline_no_diag() {
  DeviceReturnTy ret1 = device_only_function(1);
  DeviceReturnTy2 ret2 = device_only_function(1.0f);
  HostReturnTy ret3 = host_only_function(1);
  HostReturnTy2 ret4 = host_only_function(1.0f);
}

// wrong-sided overloading should cause diagnostic if it is emitted.
// This inline function is emitted since it is called by an emitted function.
inline __host__ __device__ void test_host_device_wrong_side_overloading_inline_diag() {
  DeviceReturnTy ret1 = device_only_function(1);
  DeviceReturnTy2 ret2 = device_only_function(1.0f);
#ifndef __CUDA_ARCH__
  // expected-error@-3 {{reference to __device__ function 'device_only_function' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __device__ function 'device_only_function' in __host__ __device__ function}}
#endif
  HostReturnTy ret3 = host_only_function(1);
  HostReturnTy2 ret4 = host_only_function(1.0f);
#ifdef __CUDA_ARCH__
  // expected-error@-3 {{reference to __host__ function 'host_only_function' in __host__ __device__ function}}
  // expected-error@-3 {{reference to __host__ function 'host_only_function' in __host__ __device__ function}}
#endif
}

__host__ __device__ void test_host_device_wrong_side_overloading_inline_diag_caller() {
  test_host_device_wrong_side_overloading_inline_diag();
  // expected-note@-1 {{called by 'test_host_device_wrong_side_overloading_inline_diag_caller'}}
}

// Verify that we allow overloading function templates.
template <typename T> __host__ T template_overload(const T &a) { return a; };
template <typename T> __device__ T template_overload(const T &a) { return a; };

__host__ void test_host_template_overload() {
  template_overload(1); // OK. Attribute-based overloading picks __host__ variant.
}
__device__ void test_device_template_overload() {
  template_overload(1); // OK. Attribute-based overloading picks __device__ variant.
}

// Two classes with `operator-` defined. One of them is device only.
struct C1;
struct C2;
__device__
int operator-(const C1 &x, const C1 &y);
int operator-(const C2 &x, const C2 &y);

template <typename T>
__host__ __device__ int constexpr_overload(const T &x, const T &y) {
  return x - y;
}

// Verify that function overloading doesn't prune candidate wrongly.
int test_constexpr_overload(C2 &x, C2 &y) {
  return constexpr_overload(x, y);
}

// Verify no ambiguity for new operator.
void *a = new int;
__device__ void *b = new int;
// expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, __shared__, and __managed__ variables}}

// Verify no ambiguity for new operator.
template<typename _Tp> _Tp&& f();
template<typename _Tp, typename = decltype(new _Tp(f<_Tp>()))>
void __test();

void foo() {
  __test<int>();
}

// Test resolving implicit host device candidate vs wrong-sided candidate.
// In device compilation, implicit host device caller choose implicit host
// device candidate and wrong-sided candidate with equal preference.
// Resolution result should not change with/without pragma.
namespace ImplicitHostDeviceVsWrongSided {
HostReturnTy callee(double x);
#pragma clang force_cuda_host_device begin
HostDeviceReturnTy callee(int x);
inline HostReturnTy implicit_hd_caller() {
  return callee(1.0);
}
#pragma clang force_cuda_host_device end
}

// Test resolving implicit host device candidate vs same-sided candidate.
// In host compilation, implicit host device caller choose implicit host
// device candidate and same-sided candidate with equal preference.
// Resolution result should not change with/without pragma.
namespace ImplicitHostDeviceVsSameSide {
HostReturnTy callee(int x);
#pragma clang force_cuda_host_device begin
HostDeviceReturnTy callee(double x);
inline HostDeviceReturnTy implicit_hd_caller() {
  return callee(1.0);
}
#pragma clang force_cuda_host_device end
}

// Test resolving explicit host device candidate vs. wrong-sided candidate.
// When -fgpu-defer-diag is off, wrong-sided candidate is not excluded, therefore
// the first callee is chosen.
// When -fgpu-defer-diag is on, wrong-sided candidate is excluded, therefore
// the second callee is chosen.
namespace ExplicitHostDeviceVsWrongSided {
HostReturnTy callee(double x);
__host__ __device__ HostDeviceReturnTy callee(int x);
#if __CUDA_ARCH__ && DEFER
typedef HostDeviceReturnTy ExpectedRetTy;
#else
typedef HostReturnTy ExpectedRetTy;
#endif
inline __host__ __device__ ExpectedRetTy explicit_hd_caller() {
  return callee(1.0);
}
}

// In the implicit host device function 'caller', the second 'callee' should be
// chosen since it has better match, even though it is an implicit host device
// function whereas the first 'callee' is a host function. A diagnostic will be
// emitted if the first 'callee' is chosen since deduced return type cannot be
// used before it is defined.
namespace ImplicitHostDeviceByConstExpr {
template <class a> a b;
auto callee(...);
template <class d> constexpr auto callee(d) -> decltype(0);
struct e {
  template <class ad, class... f> static auto g(ad, f...) {
    return h<e, decltype(b<f>)...>;
  }
  struct i {
    template <class, class... f> static constexpr auto caller(f... k) {
      return callee(k...);
    }
  };
  template <class, class... f> static auto h() {
    return i::caller<int, f...>;
  }
};
class l {
  l() {
    e::g([] {}, this);
  }
};
}

// Implicit HD candidate competes with device candidate.
// a and b have implicit HD copy ctor. In copy ctor of b, ctor of a is resolved.
// copy ctor of a should win over a(short), otherwise there will be ambiguity
// due to conversion operator.
namespace TestImplicitHDWithD {
  struct a {
    __device__ a(short);
    __device__ operator unsigned() const;
    __device__ operator int() const;
  };
  struct b {
    a d;
  };
  void f(b g) { b e = g; }
}

// Implicit HD candidate competes with host candidate.
// a and b have implicit HD copy ctor. In copy ctor of b, ctor of a is resolved.
// copy ctor of a should win over a(short), otherwise there will be ambiguity
// due to conversion operator.
namespace TestImplicitHDWithH {
  struct a {
    a(short);
    __device__ operator unsigned() const;
    __device__ operator int() const;
  };
  struct b {
    a d;
  };
  void f(b g) { b e = g; }
}

// Implicit HD candidate competes with HD candidate.
// a and b have implicit HD copy ctor. In copy ctor of b, ctor of a is resolved.
// copy ctor of a should win over a(short), otherwise there will be ambiguity
// due to conversion operator.
namespace TestImplicitHDWithHD {
  struct a {
    __host__ __device__ a(short);
    __device__ operator unsigned() const;
    __device__ operator int() const;
  };
  struct b {
    a d;
  };
  void f(b g) { b e = g; }
}

// HD candidate competes with H candidate.
// HD has type mismatch whereas H has type match.
// In device compilation, H wins when -fgpu-defer-diag is off and HD wins
// when -fgpu-defer-diags is on. In both cases the diagnostic should be
// deferred.
namespace TestDeferNoMatchingFuncNotEmitted {
  template <typename> struct a {};
  namespace b {
    struct c : a<int> {};
    template <typename d> void ag(d);
  } // namespace b
  template <typename ae>
  __host__ __device__ void ag(a<ae>) {
    ae e;
    ag(e);
  }
  void f() { (void)ag<b::c>; }
}

namespace TestDeferNoMatchingFuncEmitted {
  template <typename> struct a {};
  namespace b {
    struct c : a<int> {};
    template <typename d> void ag(d);
    // devnodeferonly-note@-1{{'ag<TestDeferNoMatchingFuncEmitted::b::c>' declared here}}
  } // namespace b
  template <typename ae>
  __host__ __device__ void ag(a<ae>) {
    ae e;
    ag(e);
    // devnodeferonly-error@-1{{reference to __host__ function 'ag<TestDeferNoMatchingFuncEmitted::b::c>' in __host__ __device__ function}}
    // devdeferonly-error@-2{{no matching function for call to 'ag'}}
    // devdeferonly-note@-3{{called by 'ag<TestDeferNoMatchingFuncEmitted::b::c>'}}
  }
  __host__ __device__ void f() { (void)ag<b::c>; }
  // devnodeferonly-note@-1{{called by 'f'}}
  // devdeferonly-note@-2{{called by 'f'}}
}

// Two HD candidates compete with H candidate.
// HDs have type mismatch whereas H has type match.
// In device compilation, H wins when -fgpu-defer-diag is off and two HD win
// when -fgpu-defer-diags is on. In both cases the diagnostic should be
// deferred.
namespace TestDeferAmbiguityNotEmitted {
  template <typename> struct a {};
  namespace b {
    struct c : a<int> {};
    template <typename d> void ag(d, int);
  } // namespace b
  template <typename ae>
  __host__ __device__ void ag(a<ae>, float) {
    ae e;
    ag(e, 1);
  }
  template <typename ae>
  __host__ __device__ void ag(a<ae>, double) {
  }
  void f() {
    b::c x;
    ag(x, 1);
  }
}

namespace TestDeferAmbiguityEmitted {
  template <typename> struct a {};
  namespace b {
    struct c : a<int> {};
    template <typename d> void ag(d, int);
    // devnodeferonly-note@-1{{'ag<TestDeferAmbiguityEmitted::b::c>' declared here}}
  } // namespace b
  template <typename ae>
  __host__ __device__ void ag(a<ae>, float) {
    // devdeferonly-note@-1{{candidate function [with ae = int]}}
    ae e;
    ag(e, 1);
  }
  template <typename ae>
  __host__ __device__ void ag(a<ae>, double) {
    // devdeferonly-note@-1{{candidate function [with ae = int]}}
  }
  __host__ __device__ void f() {
    b::c x;
    ag(x, 1);
    // devnodeferonly-error@-1{{reference to __host__ function 'ag<TestDeferAmbiguityEmitted::b::c>' in __host__ __device__ function}}
    // devdeferonly-error@-2{{call to 'ag' is ambiguous}}
  }
}

// Implicit HD functions compute with H function and D function.
// In host compilation, foo(0.0, 2) should resolve to X::foo<double, int>.
// In device compilation, foo(0.0, 2) should resolve to foo(double, int).
// In either case there should be no ambiguity.
namespace TestImplicitHDWithHAndD {
  namespace X {
    inline double foo(double, double) { return 0;}
    inline constexpr float foo(float, float) { return 1;}
    inline constexpr long double foo(long double, long double) { return 2;}
    template<typename _Tp, typename _Up> inline constexpr double foo(_Tp, _Up) { return 3;}
  };
  using X::foo;
  inline __device__ double foo(double, double) { return 4;}
  inline __device__ float foo(float, int) { return 5;}
  inline __device__ float foo(int, int) { return 6;}
  inline __device__ double foo(double, int) { return 7;}
  inline __device__ float foo(float, float) { return 9;}
  template<typename _Tp, typename _Up> inline __device__ double foo(_Tp, _Up) { return 10;}

  int g() {
    return [](){
    return foo(0.0, 2);
    }();
  }
}
