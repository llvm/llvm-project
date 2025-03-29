// REQUIRES: amdgpu-registered-target
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-host-only -pedantic -nogpuinc -nogpulib -nobuiltininc -fsyntax-only -Xclang -verify %s
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-device-only -pedantic -nogpuinc -nogpulib -nobuiltininc -fsyntax-only -Xclang -verify %s

// Test that deprecation warnings for the wavefront size macro are emitted properly.

#define WRAPPED __AMDGCN_WAVEFRONT_SIZE__

#define DOUBLE_WRAPPED (WRAPPED)

template <bool C, class T = void> struct my_enable_if {};

template <class T> struct my_enable_if<true, T> {
  typedef T type;
};

__attribute__((host, device)) void use(int, const char*);

template<int N> __attribute__((host, device)) int templatify(int x) {
    return x + N;
}

__attribute__((device)) const int GlobalConst = __AMDGCN_WAVEFRONT_SIZE__; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
constexpr int GlobalConstExpr = __AMDGCN_WAVEFRONT_SIZE__; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}

#if defined(__HIP_DEVICE_COMPILE__) && (__AMDGCN_WAVEFRONT_SIZE__ == 64) // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
int foo(void);
#endif

__attribute__((device)) int device_var = __AMDGCN_WAVEFRONT_SIZE__; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}

__attribute__((device))
void device_fun() {
    use(__AMDGCN_WAVEFRONT_SIZE, "device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE' has been marked as deprecated}}
    use(__AMDGCN_WAVEFRONT_SIZE__, "device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(WRAPPED, "device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(DOUBLE_WRAPPED, "device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(GlobalConst, "device function");
    use(GlobalConstExpr, "device function");
}

__attribute__((global))
void global_fun() {
    // no warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE, "global function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE' has been marked as deprecated}}
    use(__AMDGCN_WAVEFRONT_SIZE__, "global function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(WRAPPED, "global function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(DOUBLE_WRAPPED, "global function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "global function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
}

int host_var = __AMDGCN_WAVEFRONT_SIZE__; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
int host_var_alt = __AMDGCN_WAVEFRONT_SIZE; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE' has been marked as deprecated}}
int host_var_wrapped = WRAPPED; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
int host_var_double_wrapped = DOUBLE_WRAPPED; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}

__attribute__((host))
void host_fun() {
    use(__AMDGCN_WAVEFRONT_SIZE, "host function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE' has been marked as deprecated}}
    use(__AMDGCN_WAVEFRONT_SIZE__, "host function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(WRAPPED, "host function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(DOUBLE_WRAPPED, "host function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(GlobalConst, "host function");
    use(GlobalConstExpr, "host function");
}

__attribute((host, device))
void host_device_fun() {
    use(__AMDGCN_WAVEFRONT_SIZE__, "host device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(WRAPPED, "host device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(DOUBLE_WRAPPED, "host device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host device function"); // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
}

template <unsigned int OuterWarpSize = __AMDGCN_WAVEFRONT_SIZE__> // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
class FunSelector {
public:
    template<unsigned int FunWarpSize = OuterWarpSize>
    __attribute__((device))
    auto fun(void)
        -> typename my_enable_if<(FunWarpSize <= __AMDGCN_WAVEFRONT_SIZE__), void>::type // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    {
        use(1, "yay!");
    }

    template<unsigned int FunWarpSize = OuterWarpSize>
    __attribute__((device))
    auto fun(void)
        -> typename my_enable_if<(FunWarpSize > __AMDGCN_WAVEFRONT_SIZE__), void>::type // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    {
        use(0, "nay!");
    }
};

__attribute__((device))
void device_fun_selector_user() {
    FunSelector<> f;
    f.fun<>();
    f.fun<1>();
    f.fun<1000>();

    my_enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE__), int>::type x = 42; // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
}

__attribute__((device)) my_enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE__), int>::type DeviceFunTemplateRet(void) { // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    return 42;
}

__attribute__((device)) int DeviceFunTemplateArg(my_enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE__), int>::type x) { // expected-warning {{macro '__AMDGCN_WAVEFRONT_SIZE__' has been marked as deprecated}}
    return x;
}

// expected-note@* 0+ {{macro marked 'deprecated' here}}
