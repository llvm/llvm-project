// REQUIRES: amdgpu-registered-target
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-host-only -pedantic -nogpuinc -nogpulib -nobuiltininc -fsyntax-only -Xclang -verify=onhost %s
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-device-only -pedantic -nogpuinc -nogpulib -nobuiltininc -fsyntax-only -Xclang -verify=ondevice %s

// ondevice-no-diagnostics

#include <type_traits>

#define WRAPPED __AMDGCN_WAVEFRONT_SIZE__

#define DOUBLE_WRAPPED (WRAPPED)

__attribute__((host, device)) void use(int, const char*);

template<int N> __attribute__((host, device)) int templatify(int x) {
    return x + N;
}

// no warning expected
#if defined(__HIP_DEVICE_COMPILE__) && (__AMDGCN_WAVEFRONT_SIZE__ == 64) && (__AMDGCN_WAVEFRONT_SIZE == 64)
int foo(void);
#endif

// no warning expected
__attribute__((device)) int device_var = __AMDGCN_WAVEFRONT_SIZE__;

__attribute__((device))
void device_fun() {
    // no warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE, "device function");
    use(__AMDGCN_WAVEFRONT_SIZE__, "device function");
    use(WRAPPED, "device function");
    use(DOUBLE_WRAPPED, "device function");
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "device function");
}

__attribute__((global))
void global_fun() {
    // no warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE, "global function");
    use(__AMDGCN_WAVEFRONT_SIZE__, "global function");
    use(WRAPPED, "global function");
    use(DOUBLE_WRAPPED, "global function");
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "global function");
}

// warning expected
int host_var = __AMDGCN_WAVEFRONT_SIZE__;  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
int host_var_alt = __AMDGCN_WAVEFRONT_SIZE;  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE is not available in a __host__ context}}
int host_var_wrapped = WRAPPED;  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
int host_var_double_wrapped = DOUBLE_WRAPPED;  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}

__attribute__((host))
void host_fun() {
    // warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE, "host function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE is not available in a __host__ context}}
    use(__AMDGCN_WAVEFRONT_SIZE__, "host function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
    use(WRAPPED, "host function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
    use(DOUBLE_WRAPPED, "host function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ context}}
}

__attribute((host, device))
void host_device_fun() {
    // warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE__, "host device function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ __device__ context}}
    use(WRAPPED, "host device function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ __device__ context}}
    use(DOUBLE_WRAPPED, "host device function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ __device__ context}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host device function");  // onhost-warning {{device-specific macro __AMDGCN_WAVEFRONT_SIZE__ is not available in a __host__ __device__ context}}
}

// Variations of this construct are used in rocPRIM and should compile without diagnostics.
template <unsigned int OuterWarpSize = __AMDGCN_WAVEFRONT_SIZE>
class FunSelector {
public:
    template<unsigned int FunWarpSize = OuterWarpSize>
    __attribute__((device))
    auto fun(void)
        -> typename std::enable_if<(FunWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        use(1, "yay!");
    }

    template<unsigned int FunWarpSize = OuterWarpSize>
    __attribute__((device))
    auto fun(void)
        -> typename std::enable_if<(FunWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
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

    std::enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE), int>::type x = 42;
}

__attribute__((device)) std::enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE), int>::type DeviceFunTemplateRet(void) {
    return 42;
}

__attribute__((device)) int DeviceFunTemplateArg(std::enable_if<(1 <= __AMDGCN_WAVEFRONT_SIZE), int>::type x) {
    return x;
}
