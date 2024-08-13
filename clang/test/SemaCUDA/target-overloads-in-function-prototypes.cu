// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify=expected,onhost %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify=expected,ondevice %s


// Tests to ensure that functions with host and device overloads in that are
// called outside of function bodies and variable initializers, e.g., in
// template arguments are resolved with respect to the declaration to which they
// belong.

// Opaque types used for tests:
struct DeviceTy {};
struct HostTy {};
struct HostDeviceTy {};
struct TemplateTy {};

struct TrueTy { static const bool value = true; };
struct FalseTy { static const bool value = false; };

// Select one of two types based on a boolean condition.
template <bool COND, typename T, typename F> struct select_type {};
template <typename T, typename F> struct select_type<true, T, F> { typedef T type; };
template <typename T, typename F> struct select_type<false, T, F> { typedef F type; };

template <bool C> struct check : public select_type<C, TrueTy, FalseTy> { };

// Check if two types are the same.
template<class T, class U> struct is_same : public FalseTy { };
template<class T> struct is_same<T, T> : public TrueTy { };

// A static assertion that fails at compile time if the expression E does not
// have type T.
#define ASSERT_HAS_TYPE(E, T) static_assert(is_same<decltype(E), T>::value);


// is_on_device() is true when called in a device context and false if called in a host context.
__attribute__((host)) constexpr bool is_on_device(void) { return false; }
__attribute__((device)) constexpr bool is_on_device(void) { return true; }


// this type depends on whether it occurs in host or device code
#define targetdep_t select_type<is_on_device(), DeviceTy, HostTy>::type

// Defines and typedefs with different values in host and device compilation.
#ifdef __CUDA_ARCH__
#define CurrentTarget DEVICE
typedef DeviceTy CurrentTargetTy;
typedef DeviceTy TemplateIfHostTy;
#else
#define CurrentTarget HOST
typedef HostTy CurrentTargetTy;
typedef TemplateTy TemplateIfHostTy;
#endif



// targetdep_t in function declarations should depend on the target of the
// declared function.
__attribute__((device)) targetdep_t decl_ret_early_device(void);
ASSERT_HAS_TYPE(decl_ret_early_device(), DeviceTy)

__attribute__((host)) targetdep_t decl_ret_early_host(void);
ASSERT_HAS_TYPE(decl_ret_early_host(), HostTy)

__attribute__((host,device)) targetdep_t decl_ret_early_host_device(void);
ASSERT_HAS_TYPE(decl_ret_early_host_device(), CurrentTargetTy)

// If the function target is specified too late and can therefore not be
// considered for overload resolution in targetdep_t, warn.
targetdep_t __attribute__((device)) decl_ret_late_device(void); // expected-warning {{target specifier has been ignored for overload resolution}}
ASSERT_HAS_TYPE(decl_ret_late_device(), HostTy)

// No warning necessary if the ignored attribute doesn't change the result.
targetdep_t __attribute__((host)) decl_ret_late_host(void);
ASSERT_HAS_TYPE(decl_ret_late_host(), HostTy)

targetdep_t __attribute__((host,device)) decl_ret_late_host_device(void); // expected-warning {{target specifier has been ignored for overload resolution}}
ASSERT_HAS_TYPE(decl_ret_late_host_device(), HostTy)

// An odd way of writing this, but it's possible.
__attribute__((device)) targetdep_t __attribute__((host)) decl_ret_early_device_late_host(void); // expected-warning {{target specifier has been ignored for overload resolution}}
ASSERT_HAS_TYPE(decl_ret_early_device_late_host(), DeviceTy)


// The same for function definitions and parameter types:
__attribute__((device)) targetdep_t ret_early_device(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

__attribute__((host)) targetdep_t ret_early_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

__attribute__((host, device)) targetdep_t ret_early_hostdevice(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_early_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

// The parameter is still after the attribute, so it needs no warning.
targetdep_t __attribute__((device)) // expected-warning {{target specifier has been ignored for overload resolution}}
ret_late_device(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_late_device({}), HostTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

targetdep_t __attribute__((host, device)) // expected-warning {{target specifier has been ignored for overload resolution}}
ret_late_hostdevice(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_late_hostdevice({}), HostTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

targetdep_t __attribute__((host)) ret_late_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

__attribute__((device)) targetdep_t __attribute__((host)) // expected-warning {{target specifier has been ignored for overload resolution}}
ret_early_device_late_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ret_early_device_late_host({}), DeviceTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

// The attribute is even later, so we can't choose the expected overload.
targetdep_t ret_verylate_device(targetdep_t x) __attribute__((device)) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ret_verylate_device({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

// It's possible to get two different wrong types:
targetdep_t __attribute__((device)) // expected-warning {{target specifier has been ignored for overload resolution}}
ret_late_device_verylate_host(targetdep_t x) __attribute__((host)) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ret_late_device_verylate_host({}), HostTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}


// Increasingly unusual ways to specify a return type:

// The attribute is specified much earlier than the overload happens, works as
// expected.
__attribute__((device)) auto autoret_early_device(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(autoret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

// The attribute is specified much earlier than the overload happens, works as
// expected.
__attribute__((host)) auto autoret_early_host(targetdep_t x) -> targetdep_t  {
  ASSERT_HAS_TYPE(autoret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

// The attribute is specified much earlier than the overload happens, works as
// expected.
__attribute__((host,device)) auto autoret_early_hostdevice(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(autoret_early_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}


// The attribute is still specified earlier than the overload happens, works as
// expected.
auto __attribute__((device)) autoret_late_device(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(autoret_late_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

// The attribute is still specified earlier than the overload happens, works as
// expected.
auto __attribute__((host)) autoret_late_host(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(autoret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

// The attribute is still specified earlier than the overload happens, works as
// expected.
auto __attribute__((host,device)) autoret_late_hostdevice(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(autoret_late_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}


// There should be no problem if the return type is inferred from an expression in the body:
auto __attribute__((device)) fullauto_device(targetdep_t x) {
  ASSERT_HAS_TYPE(x, DeviceTy)
  return (targetdep_t)(x);
}
ASSERT_HAS_TYPE(fullauto_device({}), DeviceTy)

auto __attribute__((host)) fullauto_host(targetdep_t x) {
  ASSERT_HAS_TYPE(x, HostTy)
  return (targetdep_t)(x);
}
ASSERT_HAS_TYPE(fullauto_host({}), HostTy)

// The return type is as expected, but the argument type precedes the attribute,
// so we don't get the right type for it.
auto fullauto_verylate_device(targetdep_t x) __attribute__((device)) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(x, HostTy)
  return targetdep_t();
}
ASSERT_HAS_TYPE(fullauto_verylate_device({}), DeviceTy)

auto fullauto_verylate_host(targetdep_t x) __attribute__((host)) {
  ASSERT_HAS_TYPE(x, HostTy)
  return targetdep_t();
}
ASSERT_HAS_TYPE(fullauto_verylate_host({}), HostTy)


// MS __declspec syntax:
__declspec(__device__) targetdep_t ms_ret_early_device(targetdep_t x) {
  ASSERT_HAS_TYPE(ms_ret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

__declspec(__host__) targetdep_t ms_ret_early_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ms_ret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

__declspec(__host__) __declspec(__device__) targetdep_t ms_ret_early_hostdevice(targetdep_t x) {
  ASSERT_HAS_TYPE(ms_ret_early_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

targetdep_t __declspec(__device__) ms_ret_late_device(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ms_ret_late_device({}), HostTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

targetdep_t __declspec(__host__) ms_ret_late_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ms_ret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

targetdep_t __declspec(__host__) __declspec(__device__) ms_ret_late_hostdevice(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ms_ret_late_hostdevice({}), HostTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

__declspec(__device__) targetdep_t __declspec(__host__) ms_ret_early_device_late_host(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ms_ret_early_device_late_host({}), DeviceTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

__declspec(__device__) auto ms_autoret_early_device(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

__declspec(__host__) auto ms_autoret_early_host(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

__declspec(__host__) __declspec(__device__) auto ms_autoret_early_hostdevice(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_early_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}


auto __declspec(__device__) ms_autoret_late_device(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_late_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

auto __declspec(__host__) ms_autoret_late_host(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

auto __declspec(__host__) __declspec(__device__) ms_autoret_late_hostdevice(targetdep_t x) -> targetdep_t {
  ASSERT_HAS_TYPE(ms_autoret_late_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}


// Class/Struct member functions:

struct MethodTests {
  __attribute__((device)) targetdep_t ret_early_device(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_device({}), DeviceTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }

  __attribute__((host)) targetdep_t ret_early_host(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  __attribute__((host,device)) targetdep_t ret_early_hostdevice(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_hostdevice({}), CurrentTargetTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }

  __attribute__((device)) auto autoret_early_device(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_device({}), DeviceTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }
  __attribute__((host)) auto autoret_early_host(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  __attribute__((host,device)) auto autoret_early_hostdevice(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_hostdevice({}), CurrentTargetTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }


  // Overloaded call happens in return type, attribute is after that.
  targetdep_t __attribute__((device)) ret_late_device(targetdep_t x) {  // expected-warning {{target specifier has been ignored for overload resolution}}
    ASSERT_HAS_TYPE(ret_late_device({}), HostTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }

  targetdep_t __attribute__((host)) ret_late_host(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_late_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  targetdep_t __attribute__((host,device)) ret_late_hostdevice(targetdep_t x) {  // expected-warning {{target specifier has been ignored for overload resolution}}
    ASSERT_HAS_TYPE(ret_late_hostdevice({}), HostTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }


  // Member declarations (tested in the 'tests' function further below):
  __attribute__((device)) targetdep_t decl_ret_early_device(void);
  __attribute__((host)) targetdep_t decl_ret_early_host(void);
  __attribute__((host,device)) targetdep_t decl_ret_early_hostdevice(void);
  targetdep_t __attribute__((device)) decl_ret_late_device(void);  // expected-warning {{target specifier has been ignored for overload resolution}}
  targetdep_t __attribute__((host)) decl_ret_late_host(void);
  targetdep_t __attribute__((host,device)) decl_ret_late_hostdevice(void);  // expected-warning {{target specifier has been ignored for overload resolution}}

  // for out of line definitions:
  __attribute__((device)) targetdep_t ool_ret_early_device(targetdep_t x);
  __attribute__((host)) targetdep_t ool_ret_early_host(targetdep_t x);
  __attribute__((host,device)) targetdep_t ool_ret_early_hostdevice(targetdep_t x);
  targetdep_t __attribute__((device)) ool_ret_late_device(targetdep_t x);  // expected-warning {{target specifier has been ignored for overload resolution}}
  targetdep_t __attribute__((host)) ool_ret_late_host(targetdep_t x);
  targetdep_t __attribute__((host,device)) ool_ret_late_hostdevice(targetdep_t x);  // expected-warning {{target specifier has been ignored for overload resolution}}

};

__attribute__((device)) targetdep_t MethodTests::ool_ret_early_device(targetdep_t x) {
  ASSERT_HAS_TYPE(ool_ret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

__attribute__((host)) targetdep_t MethodTests::ool_ret_early_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ool_ret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

__attribute__((host,device)) targetdep_t MethodTests::ool_ret_early_hostdevice(targetdep_t x) {
  ASSERT_HAS_TYPE(ool_ret_early_hostdevice({}), CurrentTargetTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}

targetdep_t __attribute__((device)) MethodTests::ool_ret_late_device(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ool_ret_late_device({}), HostTy)
  ASSERT_HAS_TYPE(x, DeviceTy)
  return {};
}

targetdep_t __attribute__((host)) MethodTests::ool_ret_late_host(targetdep_t x) {
  ASSERT_HAS_TYPE(ool_ret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(x, HostTy)
  return {};
}

targetdep_t __attribute__((host,device)) MethodTests::ool_ret_late_hostdevice(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
  ASSERT_HAS_TYPE(ool_ret_late_hostdevice({}), HostTy)
  ASSERT_HAS_TYPE(x, CurrentTargetTy)
  return {};
}


// members of templated structs should also work.
template <unsigned int N>
struct TemplateMethodTests {
  __attribute__((device)) targetdep_t ret_early_device(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_device({}), DeviceTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }

  __attribute__((host)) targetdep_t ret_early_host(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  __attribute__((host,device)) targetdep_t ret_early_hostdevice(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_early_hostdevice({}), CurrentTargetTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }

  __attribute__((device)) auto autoret_early_device(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_device({}), DeviceTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }

  __attribute__((host)) auto autoret_early_host(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  __attribute__((host,device)) auto autoret_early_hostdevice(targetdep_t x) -> targetdep_t {
    ASSERT_HAS_TYPE(autoret_early_hostdevice({}), CurrentTargetTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }

  targetdep_t __attribute__((device)) ret_late_device(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
    ASSERT_HAS_TYPE(ret_late_device({}), HostTy)
    ASSERT_HAS_TYPE(x, DeviceTy)
    return {};
  }

  targetdep_t __attribute__((host)) ret_late_host(targetdep_t x) {
    ASSERT_HAS_TYPE(ret_late_host({}), HostTy)
    ASSERT_HAS_TYPE(x, HostTy)
    return {};
  }

  targetdep_t __attribute__((host,device)) ret_late_hostdevice(targetdep_t x) { // expected-warning {{target specifier has been ignored for overload resolution}}
    ASSERT_HAS_TYPE(ret_late_hostdevice({}), HostTy)
    ASSERT_HAS_TYPE(x, CurrentTargetTy)
    return {};
  }


  __attribute__((device)) targetdep_t decl_ret_early_device(void);
  __attribute__((host)) targetdep_t decl_ret_early_host(void);
  __attribute__((host,device)) targetdep_t decl_ret_early_hostdevice(void);

  targetdep_t __attribute__((device)) decl_ret_late_device(void); // expected-warning {{target specifier has been ignored for overload resolution}}
  targetdep_t __attribute__((host)) decl_ret_late_host(void);
  targetdep_t __attribute__((host,device)) decl_ret_late_hostdevice(void); // expected-warning {{target specifier has been ignored for overload resolution}}
};

void tests(void) {
  MethodTests mt;

  ASSERT_HAS_TYPE(mt.ret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(mt.ret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(mt.ret_early_hostdevice({}), CurrentTargetTy)

  ASSERT_HAS_TYPE(mt.autoret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(mt.autoret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(mt.autoret_early_hostdevice({}), CurrentTargetTy)

  // The target specifier is too late to be considered:
  ASSERT_HAS_TYPE(mt.ret_late_device({}), HostTy)
  ASSERT_HAS_TYPE(mt.ret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(mt.ret_late_hostdevice({}), HostTy)

  ASSERT_HAS_TYPE(mt.decl_ret_early_device(), DeviceTy)
  ASSERT_HAS_TYPE(mt.decl_ret_early_host(), HostTy)
  ASSERT_HAS_TYPE(mt.decl_ret_early_hostdevice(), CurrentTargetTy)

  // The target specifier is too late to be considered:
  ASSERT_HAS_TYPE(mt.decl_ret_late_device(), HostTy)
  ASSERT_HAS_TYPE(mt.decl_ret_late_host(), HostTy)
  ASSERT_HAS_TYPE(mt.decl_ret_late_hostdevice(), HostTy)

  TemplateMethodTests<42> tmt;
  ASSERT_HAS_TYPE(tmt.ret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(tmt.ret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(tmt.ret_early_hostdevice({}), CurrentTargetTy)

  ASSERT_HAS_TYPE(tmt.autoret_early_device({}), DeviceTy)
  ASSERT_HAS_TYPE(tmt.autoret_early_host({}), HostTy)
  ASSERT_HAS_TYPE(tmt.autoret_early_hostdevice({}), CurrentTargetTy)

  ASSERT_HAS_TYPE(tmt.ret_late_device({}), HostTy)
  ASSERT_HAS_TYPE(tmt.ret_late_host({}), HostTy)
  ASSERT_HAS_TYPE(tmt.ret_late_hostdevice({}), HostTy)

  ASSERT_HAS_TYPE(tmt.decl_ret_early_device(), DeviceTy)
  ASSERT_HAS_TYPE(tmt.decl_ret_early_host(), HostTy)
  ASSERT_HAS_TYPE(tmt.decl_ret_early_hostdevice(), CurrentTargetTy)

  ASSERT_HAS_TYPE(tmt.decl_ret_late_device(), HostTy)
  ASSERT_HAS_TYPE(tmt.decl_ret_late_host(), HostTy)
  ASSERT_HAS_TYPE(tmt.decl_ret_late_hostdevice(), HostTy)
}


// global variables:
__attribute__((device)) targetdep_t var_early_device = {};
ASSERT_HAS_TYPE(var_early_device, DeviceTy)

targetdep_t var_early_host = {};
ASSERT_HAS_TYPE(var_early_host, HostTy)

targetdep_t __attribute__((device)) var_late_device = {}; // expected-warning {{target specifier has been ignored for overload resolution}}
ASSERT_HAS_TYPE(var_late_device, HostTy)


// Tests for the overload candidate ordering compared to templates:

enum Candidate {
  TEMPLATE,
  HOST,
  DEVICE,
  HOSTDEVICE,
};

// (1.) If the overloaded functions are constexpr

// (1.a) Prefer fitting overloads.
template <typename T> constexpr Candidate ce_template_vs_H_D_functions(T arg) { return TEMPLATE; }
__attribute__((device)) constexpr Candidate ce_template_vs_H_D_functions(float arg) { return DEVICE; }
__attribute__((host)) constexpr Candidate ce_template_vs_H_D_functions(float arg) { return HOST; }

__attribute__((device)) check<ce_template_vs_H_D_functions(1.0f) == DEVICE>::type
test_ce_template_vs_H_D_functions_for_device() {
  return TrueTy();
}

__attribute__((host)) check<ce_template_vs_H_D_functions(1.0f) == HOST>::type
test_ce_template_vs_H_D_functions_for_host() {
  return TrueTy();
}

__attribute__((host,device)) check<ce_template_vs_H_D_functions(1.0f) == CurrentTarget>::type
test_ce_template_vs_H_D_functions_for_hd() {
  return TrueTy();
}


// (1.b) Always prefer an HD candidate over a template candidate.
template <typename T> constexpr Candidate ce_template_vs_HD_function(T arg) { return TEMPLATE; }
__attribute__((host, device)) constexpr Candidate ce_template_vs_HD_function(float arg) { return HOSTDEVICE; }

__attribute__((device)) check<ce_template_vs_HD_function(1.0f) == HOSTDEVICE>::type
test_ce_template_vs_HD_function_for_device() {
  return TrueTy();
}

__attribute__((host)) check<ce_template_vs_HD_function(1.0f) == HOSTDEVICE>::type
test_ce_template_vs_HD_function_for_host() {
  return TrueTy();
}

__attribute__((host,device)) check<ce_template_vs_HD_function(1.0f) == HOSTDEVICE>::type
test_ce_template_vs_HD_function_for_hd() {
  return TrueTy();
}


// (1.c) Even wrong-sided calls are okay if the called function is constexpr, so
// prefer the device overload over the template.
template <typename T> constexpr Candidate ce_template_vs_D_function(T arg) { return TEMPLATE; }
__attribute__((device)) constexpr Candidate ce_template_vs_D_function(float arg) { return DEVICE; }

__attribute__((host)) check<ce_template_vs_D_function(1.0f) == DEVICE>::type
test_ce_template_vs_D_function_for_host() {
  return TrueTy();
}

__attribute__((device)) check<ce_template_vs_D_function(1.0f) == DEVICE>::type
test_ce_template_vs_D_function_for_device() {
  return TrueTy();
}

__attribute__((host,device)) check<ce_template_vs_D_function(1.0f) == DEVICE>::type
test_ce_template_vs_D_function_for_hd() {
  return TrueTy();
}


// (2.) If the overloaded functions are NOT constexpr

// (2.a) Prefer fitting overloads.
template <typename T> TemplateTy template_vs_H_D_functions(T arg) { return {}; }
__attribute__((device)) DeviceTy template_vs_H_D_functions(float arg) { return {}; }
__attribute__((host)) HostTy template_vs_H_D_functions(float arg) { return {}; }

__attribute__((device)) check<is_same<decltype(template_vs_H_D_functions(1.0f)), DeviceTy>::value>::type
test_template_vs_H_D_functions_for_device() {
  return TrueTy{};
}

__attribute__((host)) check<is_same<decltype(template_vs_H_D_functions(1.0f)), HostTy>::value>::type
test_template_vs_H_D_functions_for_host() {
  return TrueTy{};
}

__attribute__((host,device)) check<is_same<decltype(template_vs_H_D_functions(1.0f)), CurrentTargetTy>::value>::type
test_template_vs_H_D_functions_for_hd() {
  return TrueTy{};
}

// (2.b) Always prefer an HD candidate over a template candidate.
template <typename T> TemplateTy template_vs_HD_function(T arg) { return {}; }
__attribute__((host,device)) HostDeviceTy template_vs_HD_function(float arg) { return {}; }

__attribute__((device)) check<is_same<decltype(template_vs_HD_function(1.0f)), HostDeviceTy>::value>::type
test_template_vs_HD_function_for_device() {
  return TrueTy{};
}

__attribute__((host)) check<is_same<decltype(template_vs_HD_function(1.0f)), HostDeviceTy>::value>::type
test_template_vs_HD_function_for_host() {
  return TrueTy{};
}

__attribute__((host,device)) check<is_same<decltype(template_vs_HD_function(1.0f)), HostDeviceTy>::value>::type
test_template_vs_HD_function_for_hd() {
  return TrueTy{};
}


// (2.c) For non-constexpr functions, prefer a sameside or native template
// function over a wrongside non-template function:
template <typename T> TemplateTy template_vs_D_function(T arg) { return {}; }
__attribute__((device)) DeviceTy template_vs_D_function(float arg) { return {}; }

__attribute__((host,device)) check<is_same<decltype(template_vs_D_function(1.0f)), TemplateIfHostTy>::value>::type
test_template_vs_D_function_for_hd() {
  return TrueTy{};
}

__attribute__((device)) check<is_same<decltype(template_vs_D_function(1.0f)), DeviceTy>::value>::type
test_template_vs_D_function_for_device() {
  return TrueTy{};
}

__attribute__((host)) check<is_same<decltype(template_vs_D_function(1.0f)), TemplateTy>::value>::type
test_template_vs_D_function_for_host() {
  return TrueTy{};
}


// If only a wrongside function is available, it is selected.
__attribute__((device)) DeviceTy only_D_function(float arg) { return {}; }

__attribute__((host)) check<is_same<decltype(only_D_function(1.0f)), DeviceTy>::value>::type
test_only_D_function_for_host() {
  return TrueTy{};
}
