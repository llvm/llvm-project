// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=expected,dev %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -fsyntax-only -verify=expected,dev %s
// RUN: %clang_cc1 -x hip -fsyntax-only -verify %s

#include "Inputs/cuda.h"

namespace std {
class type_info {};
} // namespace std

struct B {
  __host__ __device__ virtual ~B() {}
};
struct D : B {};

void host(B *b) {
  (void)dynamic_cast<D *>(b);
  (void)typeid(*b);
}

__host__ void explicit_host(B *b) {
  (void)dynamic_cast<D *>(b);
  (void)typeid(*b);
}

__device__ void device(B *b, D *d) {
  (void)dynamic_cast<D *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  (void)dynamic_cast<D &>(*b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  (void)typeid(D);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  (void)typeid(*b);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  (void)sizeof(typeid(int));
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  (void)sizeof(dynamic_cast<D *>(b));
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  decltype(dynamic_cast<D *>(b)) ptr = d;
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  (void)ptr;

  // As with -fno-rtti, these don't use RTTI and are allowed.
  (void)dynamic_cast<void *>(b);
  (void)dynamic_cast<B *>(d);
}

__global__ void kernel(B *b) {
  (void)dynamic_cast<D *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __global__ function as RTTI is not available in device code}}
  (void)typeid(*b);
  // expected-error@-1 {{cannot use 'typeid' in __global__ function as RTTI is not available in device code}}
}

struct S {
  __device__ D *member(B *b) { return dynamic_cast<D *>(b); }
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
};

// A __device__ lambda is device code, and an unannotated lambda is
// __host__ __device__, so it is device code when called from device code.
__device__ void lambdas(B *b) {
  auto dev = [] __device__ (B *p) { return dynamic_cast<D *>(p); };
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  auto hd = [](B *p) { return dynamic_cast<D *>(p); };
  // dev-error@-1 {{cannot use 'dynamic_cast' in __host__ __device__ function as RTTI is not available in device code}}
  dev(b);
  hd(b);
  // dev-note@-1 {{called by 'lambdas'}}
}

// Check that it's an error to use RTTI from a __host__ __device__ function if
// and only if it's codegen'ed for device.

__host__ __device__ void hd1(B *b) {
  (void)dynamic_cast<D *>(b);
  // dev-error@-1 {{cannot use 'dynamic_cast' in __host__ __device__ function as RTTI is not available in device code}}
}

// No error, never instantiated on device.
inline __host__ __device__ void hd2(B *b) { (void)typeid(*b); }
void call_hd2(B *b) { hd2(b); }

// Error, instantiated on device.
inline __host__ __device__ void hd3(B *b) {
  (void)typeid(*b);
  // dev-error@-1 {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
}
__device__ void call_hd3(B *b) { hd3(b); }
// dev-note@-1 {{called by 'call_hd3'}}

// Templates are checked when they are instantiated.
template <class T> __device__ T *tmpl_unused(B *b) {
  return dynamic_cast<T *>(b);
}

template <class T> __device__ T *tmpl(B *b) {
  return dynamic_cast<T *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl(B *b) { tmpl<D>(b); }
// expected-note@-1 {{in instantiation of function template specialization 'tmpl<D>' requested here}}

template <class T> __device__ void tmpl_typeid(T *t) {
  (void)typeid(*t);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl_typeid(B *b) { tmpl_typeid(b); }
// expected-note@-1 {{in instantiation of function template specialization 'tmpl_typeid<B>' requested here}}

template <class T> __device__ void tmpl_typeid_unused(T *t) {
  (void)typeid(*t);
}

// A non-dependent operand is checked once, in the template definition.
template <class T> __device__ void tmpl_nondependent() {
  (void)typeid(int);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl_nondependent() { tmpl_nondependent<int>(); }

template <class T> __device__ T *explicit_inst(B *b) {
  return dynamic_cast<T *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
}
template __device__ D *explicit_inst<D>(B *);
// expected-note@-1 {{in instantiation of function template specialization 'explicit_inst<D>' requested here}}

template <class T> __global__ void kernel_tmpl(B *b) {
  (void)typeid(*static_cast<T *>(b));
  // expected-error@-1 {{cannot use 'typeid' in __global__ function as RTTI is not available in device code}}
}
template __global__ void kernel_tmpl<D>(B *);
// expected-note@-1 {{in instantiation of function template specialization 'kernel_tmpl<D>' requested here}}

// A host virtual function in a class that also has device virtual functions
// is not device code.
struct Fix {
  __device__ virtual void run() {}
  virtual D *init(B *b) { return dynamic_cast<D *>(b); }
};
__device__ void use_fix() { Fix f; f.run(); }

// Outside a function, the initializer of a device variable is device code.
__device__ const std::type_info *device_var = &typeid(int);
// expected-error@-1 {{cannot use 'typeid' in the initializer of a device variable as RTTI is not available in device code}}
__constant__ const std::type_info *constant_var = &typeid(int);
// expected-error@-1 {{cannot use 'typeid' in the initializer of a device variable as RTTI is not available in device code}}
const std::type_info *host_var = &typeid(int);

// A default member initializer is checked where it is used.
struct MemberInit { // #member_init_struct
  const std::type_info *t = &typeid(int); // #member_init
};
// expected-error@#member_init {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
// dev-error@#member_init {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
// dev-note@#member_init_struct {{default member initializer used here}}

__device__ void member_init_ctor() {
  MemberInit m;
  // dev-note@-1 {{called by 'member_init_ctor'}}
  (void)m;
}
__device__ void member_init_aggregate() {
  MemberInit m{};
  // expected-note@-1 {{default member initializer used here}}
  (void)m;
}
void member_init_host() {
  MemberInit m;
  MemberInit n{};
  (void)m;
  (void)n;
}

struct MemberInitDeviceCtor {
  B *b = nullptr;
  D *d = dynamic_cast<D *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  __device__ MemberInitDeviceCtor() {}
  // expected-note@-1 {{default member initializer used here}}
};

// No error, the constructor is never used on device.
struct MemberInitHDCtor {
  const std::type_info *t = &typeid(int);
  __host__ __device__ MemberInitHDCtor() {}
};
void member_init_hd_ctor_host() { MemberInitHDCtor m; }

template <class T> struct MemberInitTmpl { // #member_init_tmpl
  const std::type_info *t = &typeid(T);
  // dev-error@-1 {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
};
// dev-note@#member_init_tmpl {{default member initializer used here}}
__device__ void member_init_tmpl() {
  MemberInitTmpl<int> m;
  // dev-note@-1 {{called by 'member_init_tmpl'}}
  (void)m;
}

// Whether a dependent default member initializer uses RTTI can depend on the
// template arguments.
template <class T> struct MemberInitDepCast { // #member_init_dep_cast
  B *b = nullptr;
  T *t = dynamic_cast<T *>(b);
  // dev-error@-1 {{cannot use 'dynamic_cast' in __host__ __device__ function as RTTI is not available in device code}}
};
// dev-note@#member_init_dep_cast {{default member initializer used here}}
__device__ void member_init_dep_cast() {
  MemberInitDepCast<D> d;
  // dev-note@-1 {{called by 'member_init_dep_cast'}}
  MemberInitDepCast<B> b;
  (void)d;
  (void)b;
}

// No error, only instantiated for host code.
template <class T> struct MemberInitTmplHost {
  const std::type_info *t = &typeid(T);
};
void member_init_tmpl_host() {
  MemberInitTmplHost<int> m;
  (void)m;
}

// A default argument is checked where it is used.
__device__ void default_arg(const std::type_info *t = &typeid(int)); // #default_arg
// expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
__device__ void use_default_arg() {
  default_arg();
  // expected-note@-1 {{default argument used here}}
}

inline __host__ __device__ void hd_default_arg(const std::type_info *t = &typeid(int)) {}
// expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
void use_hd_default_arg_host() { hd_default_arg(); }
__device__ void use_hd_default_arg_device() {
  hd_default_arg();
  // expected-note@-1 {{default argument used here}}
}

// Diagnosed once, at the use, although the default argument is parsed in a
// __device__ function.
__device__ void local_default_arg() {
  __device__ void local(const std::type_info *t = &typeid(int));
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  local();
  // expected-note@-1 {{default argument used here}}
}

// A dependent default argument is checked when it is instantiated for a use.
template <class T>
__device__ void dep_default_arg(const std::type_info *t = &typeid(T)); // #dep_default_arg
// expected-error@#dep_default_arg {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
__device__ void use_dep_default_arg() {
  dep_default_arg<int>();
  // expected-note@-1 {{default argument used here}}
  // No error, the default argument is not used.
  dep_default_arg<float>(nullptr);
}

// Whether a dependent default argument uses RTTI can depend on the template
// arguments.
template <class T>
__device__ void dep_cast_default_arg(T *p = dynamic_cast<T *>(static_cast<B *>(nullptr)));
// expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
__device__ void use_dep_cast_default_arg() {
  dep_cast_default_arg<D>();
  // expected-note@-1 {{default argument used here}}
  dep_cast_default_arg<B>();
  dep_cast_default_arg<void>();
}

template <class T> struct DepDefaultArgMember {
  __device__ void f(const std::type_info *t = &typeid(T));
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
};
__device__ void use_dep_default_arg_member() {
  DepDefaultArgMember<int>().f();
  // expected-note@-1 {{default argument used here}}
}

template <class T>
inline __host__ __device__ void hd_dep_default_arg(const std::type_info *t = &typeid(T)) {}
// expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
void use_hd_dep_default_arg_host() { hd_dep_default_arg<int>(); }
__device__ void use_hd_dep_default_arg_device() {
  hd_dep_default_arg<float>();
  // expected-note@-1 {{default argument used here}}
}

// A non-dependent default argument of a template is checked for each
// specialization that uses it.
template <class T>
__device__ void nondep_default_arg(T, const std::type_info *t = &typeid(int)); // #nondep_default_arg
// expected-error@#nondep_default_arg 2{{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
__device__ void use_nondep_default_arg() {
  nondep_default_arg(1);
  // expected-note@-1 {{default argument used here}}
  nondep_default_arg(1.0);
  // expected-note@-1 {{default argument used here}}
  nondep_default_arg(2);
}

// A default argument used in a default member initializer.
struct NestedDefaultArg { // #nested_struct
  const std::type_info *t = (default_arg(), nullptr);
};
// dev-error@#default_arg {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
// dev-note@#nested_struct {{default member initializer used here}}
__device__ void nested_default_arg() {
  NestedDefaultArg n;
  // dev-note@-1 {{called by 'nested_default_arg'}}
  (void)n;
}

// A dependent default argument used in a dependent default member initializer.
template <class T> struct DepNestedDefaultArg { // #dep_nested_struct
  const std::type_info *t = (dep_default_arg<T>(), nullptr);
};
// dev-error@#dep_default_arg {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
// dev-note@#dep_nested_struct {{default member initializer used here}}
__device__ void dep_nested_default_arg() {
  DepNestedDefaultArg<long> n;
  // dev-note@-1 {{called by 'dep_nested_default_arg'}}
  (void)n;
}

// A lambda body is checked as a function of its own, while the initializer of
// a capture is part of the default member initializer.
struct LambdaBody {
  const std::type_info *t = [] { return &typeid(int); }(); // #lambda_body
};
// dev-error@#lambda_body {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
// dev-note@#lambda_body {{called by 'lambda_body'}}
__device__ void lambda_body() {
  LambdaBody l;
  (void)l;
}

struct LambdaCapture { // #lambda_capture_struct
  const std::type_info *t = [p = &typeid(int)] { return p; }();
  // dev-error@-1 {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
};
// dev-note@#lambda_capture_struct {{default member initializer used here}}
__device__ void lambda_capture() {
  LambdaCapture l;
  // dev-note@-1 {{called by 'lambda_capture'}}
  (void)l;
}
