// REQUIRES: amdgpu-registered-target
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-host-only -pedantic -nogpuinc -nogpulib -nobuiltininc -nostdinc -fsyntax-only -Xclang -verify=onhost %s
// RUN: %clang -xhip --offload-arch=gfx1030 --offload-device-only -pedantic -nogpuinc -nogpulib -nobuiltininc -nostdinc -fsyntax-only -Xclang -verify=ondevice %s

// ondevice-no-diagnostics

#define WRAPPED __AMDGCN_WAVEFRONT_SIZE__

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
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "device function");
}

// warning expected
int host_var = __AMDGCN_WAVEFRONT_SIZE__;  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}

__attribute__((host))
void host_fun() {
    // warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE, "host function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
    use(__AMDGCN_WAVEFRONT_SIZE__, "host function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
    use(WRAPPED, "host function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
}

__attribute((host, device))
void host_device_fun() {
    // warnings expected
    use(__AMDGCN_WAVEFRONT_SIZE__, "host device function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
    use(WRAPPED, "host device function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
    use(templatify<__AMDGCN_WAVEFRONT_SIZE__>(42), "host device function");  // onhost-warning {{'__hip_device_macro_guard' is deprecated: The __AMDGCN_WAVEFRONT_SIZE macros do not correspond to the device(s) when used in host code and may only be used in device code.}}
}

// onhost-note@__clang_hip_device_macro_guards.h:45 0+ {{'__hip_device_macro_guard' has been explicitly marked deprecated here}}
