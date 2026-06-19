---
myst:
    html_meta:
        "description": "Describes using the LLVM ASan on a GPU"
        "keywords": "LLVM, ASan, address sanitizer, AddressSanitizer, instrumented libraries, instrumented applications, AMD, ROCm"
---

# Using the AddressSanitizer on a GPU

The LLVM AddressSanitizer (ASan) provides a process that allows developers to detect runtime addressing errors in applications and libraries. The detection is achieved using a combination of compiler-added instrumentation and runtime techniques, including function interception and replacement.
Until now, the LLVM ASan process was only available for traditional purely CPU applications. However, ROCm has extended this mechanism to additionally allow the detection of some addressing errors on the GPU in heterogeneous applications. Ideally, developers should treat heterogeneous HIP and OpenMP applications exactly like pure CPU applications. However, this simplicity has not been achieved yet.
This document describes using ROCm ASan.

For information about LLVM ASan, see the [LLVM documentation](https://clang.llvm.org/docs/AddressSanitizer.html).

:::{note}
LLVM ASan for ROCm is tested and supported on Ubuntu 22.04, RHEL 8, and SLES 15. 
:::

## GPU Address Sanitizer release updates (ASan)

Changes were added to santizer_common and ASan libraries in compiler-rt to support AMD GPU address sanitizer error detection and reports. These changes are intended to be upstreamed. The instrumentation pass changes have already been upstreamed.

## Compiling for ASan

The ASan process begins by compiling the application of interest with the ASan instrumentation.

Recommendations for doing this are:

* Compile as many application and dependent library sources as possible using an AMD-built clang-based compiler such as `amdclang++`.
* Add the following options to the existing compiler and linker options:
  
  * `-fsanitize=address` - enables instrumentation

  * `-shared-libsan` - use shared version of runtime

  * `-g` - add debug info for improved reporting

* Explicitly use `xnack+` in the offload architecture option. For example, `--offload-arch=gfx90a:xnack+`

Other architectures are allowed, but their device code will not be instrumented and a warning will be emitted.

:::{tip}
It is not an error to compile some files without ASan instrumentation, but doing so reduces the ability of the process to detect addressing errors. However, if the main program "`a.out`" does not directly depend on the ASan runtime (`libclang_rt.asan-x86_64.so`) after the build completes (check by running `ldd` (List Dynamic Dependencies) or `readelf`), the application will immediately report an error at runtime as described in the next section.
:::

:::{note}
When compiling OpenMP programs with ASan instrumentation, it is currently necessary to set the environment variable `LIBRARY_PATH` to `/opt/rocm-<version>/lib/llvm/lib/asan:/opt/rocm-<version>/lib/asan`. At runtime, it may be necessary to add `/opt/rocm-<version>/lib/llvm/lib/asan` to `LD_LIBRARY_PATH`.
:::

### About compilation time

When `-fsanitize=address` is used, the LLVM compiler adds instrumentation code around every memory operation. This added code must be handled by all downstream components of the compiler toolchain and results in increased overall compilation time. This increase is especially evident in the AMDGPU device compiler and has in a few instances raised the compile time to an unacceptable level.

There are a few options if the compile time becomes unacceptable:

* Avoid instrumentation of the files which have the worst compile times. This will reduce the effectiveness of the ASan process.
* Add the option `-fsanitize-recover=address` to the compiles with the worst compile times. This option simplifies the added instrumentation resulting in faster compilation. See below for more information.
* Disable instrumentation on a per-function basis by adding `__attribute__`((no_sanitize("address"))) to functions found to be responsible for the large compile time. Again, this will reduce the effectiveness of the process.

## Installing ROCm GPU ASan packages

For a complete ROCm GPU Sanitizer installation, including packages, instrumented HSA and HIP runtimes, tools, and math libraries, use the following command as described in each install instructions for your operating systems under [Installation via AMDGPU installer](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer-index.html):

```bash
    sudo amdgpu-install -usecase=rocm,asan
```

## Using AMD-supplied ASan instrumented libraries

ROCm releases have optional packages that contain additional ASan instrumented builds of the ROCm libraries (usually found in `/opt/rocm-<version>/lib`). The instrumented libraries have identical names to the regular uninstrumented libraries, and are located in `/opt/rocm-<version>/lib/asan`.
These additional libraries are built using the `amdclang++` and `hipcc` compilers, while some uninstrumented libraries are built with `g++`. The preexisting build options are used but, as described above, additional options are used: `-fsanitize=address`, `-shared-libsan` and `-g`.

These additional libraries avoid additional developer effort to locate repositories, identify the correct branch, check out the correct tags, and other efforts needed to build the libraries from the source. And they extend the ability of the process to detect addressing errors into the ROCm libraries themselves.

When adjusting an application build to add instrumentation, linking against these instrumented libraries is unnecessary. For example, any `-L` `/opt/rocm-<version>/lib` compiler options need not be changed. However, the instrumented libraries should be used when the application is run. It is particularly important that the instrumented language runtimes, like `libamdhip64.so` and `librocm-core.so`, are used; otherwise, device invalid access detections may not be reported.

## Running ASan instrumented applications

### Preparing to run an instrumented application

Here are a few recommendations to consider before running an ASan instrumented heterogeneous application.

* Ensure the Linux kernel running on the system has Heterogeneous Memory Management (HMM) support. A kernel version of 5.6 or higher should be sufficient.
* Ensure XNACK is enabled
  * For `gfx90a` (MI-2X0) or `gfx940` (MI-3X0) use environment `HSA_XNACK = 1`.
  * For `gfx906` (MI-50) or `gfx908` (MI-100) use environment `HSA_XNACK = 1` but also ensure the amdgpu kernel module is loaded with module argument `noretry=0`.
This requirement is due to the fact that the XNACK setting for these GPUs is system-wide.

* Ensure that the application will use the instrumented libraries when it runs. The output from the shell command `ldd <application name>` can be used to see which libraries will be used.
If the instrumented libraries are not listed by `ldd`, the environment variable `LD_LIBRARY_PATH` may need to be adjusted, or in some cases an `RPATH` compiled into the application may need to be changed and the application recompiled.

* Ensure that the application depends on the ASan runtime. This can be checked by running the command `readelf -d <application name> | grep NEEDED` and verifying that shared library: `libclang_rt.asan-x86_64.so` appears in the output.
If it does not appear, when executed the application will quickly output an ASan error that looks like:

```bash
==3210==ASan runtime does not come first in initial library list; you should either link runtime to your application or manually preload it with LD_PRELOAD.
```

* Ensure that the application `llvm-symbolizer` can be executed, and that it is located in `/opt/rocm-<version>/llvm/bin`. This executable is not strictly required, but if found is used to translate ("symbolize") a host-side instruction address into a more useful function name, file name, and line number (assuming the application has been built to include debug information).

There is an environment variable, `ASAN_OPTIONS`, that can be used to adjust the runtime behavior of the ASan runtime itself. There are more than a hundred "flags" that can be adjusted (see an old list at [flags](https://github.com/google/sanitizers/wiki/AddressSanitizerFlags)) but the default settings are correct and should be used in most cases. It must be noted that these options only affect the host ASan runtime. The device runtime only currently supports the default settings for the few relevant options.

There are three `ASAN_OPTION` flags of note.

* `halt_on_error=0/1 default 1`.

  This tells the ASan runtime to halt the application immediately after detecting and reporting an addressing error. The default makes sense because the application has entered the realm of undefined behavior. If the developer wishes to have the application continue anyway, this option can be set to zero. However, the application and libraries should then be compiled with the additional option `-fsanitize-recover=address`. Note that the ROCm optional ASan instrumented libraries are not compiled with this option and if an error is detected within one of them, but halt_on_error is set to 0, more undefined behavior will occur.

* `detect_leaks=0/1 default 1`.

  This option directs the ASan runtime to enable the [Leak Sanitizer](https://clang.llvm.org/docs/LeakSanitizer.html) (LSan). For heterogeneous applications, this default results in significant output from the leak sanitizer when the application exits due to allocations made by the language runtime which are not considered to be leaks. This output can be avoided by adding `detect_leaks=0` to the `ASAN_OPTIONS`, or alternatively by producing an LSan suppression file (syntax described [here](https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer)) and activating it with environment variable `LSAN_OPTIONS=suppressions=/path/to/suppression/file`. When using a suppression file, a suppression report is printed by default. The suppression report can be disabled by using the `LSAN_OPTIONS` flag `print_suppressions=0`.

* `quarantine_size_mb=N default 256`

  This option defines the number of megabytes (MB) `N` of memory that the ASan runtime will hold after it is `freed` to detect use-after-free situations. This memory is unavailable for other purposes. The default of 256 MB may be too small to detect some use-after-free situations, especially given that the large size of many GPU memory allocations may push `freed` allocations out of quarantine before the attempted use.

  :::{note}
  Setting the value of `quarantine_size_mb` larger may enable more problematic uses to be detected, but at the cost of reducing memory available for other purposes.
  :::

## Runtime overhead

Running an ASan instrumented application incurs
overheads which may result in unacceptably long runtimes
or failure to run at all.

### Higher execution time

ASan detection works by checking each address at runtime
before the address is actually accessed by a load, store, or atomic
instruction.
This checking involves an additional load to "shadow" memory which
records whether the address is "poisoned" or not, and additional logic
that decides whether to produce an detection report or not.

This extra runtime work can cause the application to slow down by
a factor of three or more, depending on how many memory accesses are
executed.
For heterogeneous applications, the shadow memory must be accessible by all devices
and this can mean that shadow accesses from some devices may be more costly
than non-shadow accesses.

### Higher memory use

The address checking described above relies on the compiler to surround
each program variable with a red zone and on ASan
runtime to surround each runtime memory allocation with a red zone and
fill the shadow corresponding to each red zone with poison.
The added memory for the red zones is additional overhead on top
of the 13% overhead for the shadow memory itself.

Applications which consume most one or more available memory pools when
run normally are likely to encounter allocation failures when run with
instrumentation.

## Runtime reporting

It is not the intention of this document to provide a detailed explanation of all the types of reports that can be output by the ASan runtime. Instead, the focus is on the differences between the standard reports for CPU issues, and reports for GPU issues.

An invalid address detection report for the CPU always starts with

```bash
==<PID>==ERROR: AddressSanitizer: <problem type> on address <memory address> at pc <pc> bp <bp> sp <sp> <access> of size <N> at <memory address> thread T0
```

and continues with a stack trace for the access, a stack trace for the allocation and deallocation, if relevant, and a dump of the shadow near the <memory address>.

In contrast, an invalid address detection report for the GPU always starts with

```bash
==<PID>==ERROR: AddressSanitizer: <problem type> on amdgpu device <device> at pc <pc> <access> of size <n> in workgroup id (<X>,<Y>,<Z>)
```

Above, `<device>` is the integer device ID, and `(<X>, <Y>, <Z>)` is the ID of the workgroup or block where the invalid address was detected.

While the CPU report include a call stack for the thread attempting the invalid access, the GPU is currently to a call stack of size one, i.e. the (symbolized) of the invalid access, e.g.

```bash
#0 <pc> in <fuction signature> at /path/to/file.hip:<line>:<column>
```

This short call stack is followed by a GPU unique section that looks like

```bash
Thread ids and accessed addresses:
<lid0> <maddr 0> : <lid1> <maddr1> : ...
```

where each `<lid j> <maddr j>` indicates the lane ID and the invalid memory address held by lane `j` of the wavefront attempting the invalid access.

Additionally, reports for invalid GPU accesses to memory allocated by GPU code via `malloc` or new starting with, for example,

```bash
==1234==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc 0x7fa9f5c92dcc
```

or

```bash
==5678==ERROR: AddressSanitizer: heap-use-after-free on amdgpu device 3 at pc 0x7f4c10062d74
```

currently may include one or two surprising CPU side tracebacks mentioning :`hostcall`". This is due to how `malloc` and `free` are implemented for GPU code and these call stacks can be ignored.

## Running ASan with `rocgdb`

`rocgdb` can be used to further investigate ASan detected errors, with some preparation.

Currently, the ASan runtime complains when starting `rocgdb` without preparation.

```bash
$ rocgdb my_app
==1122==ASan` runtime does not come first in initial library list; you should either link runtime to your application or manually preload it with LD_PRELOAD.
```

This is solved by setting environment variable `LD_PRELOAD` to the path to the ASan runtime, whose path can be obtained using the command

```bash
amdclang++ -print-file-name=libclang_rt.asan-x86_64.so
```

You should also set the environment variable `HIP_ENABLE_DEFERRED_LOADING=0` before debugging HIP applications.

After starting `rocgdb` breakpoints can be set on the ASan runtime error reporting entry points of interest. For example, if an ASan error report includes

```bash
WRITE of size 4 in workgroup id (10,0,0)
```

the `rocgdb` command needed to stop the program before the report is printed is

```bash
(gdb) break __asan_report_store4
```

Similarly, the appropriate command for a report including

```bash
READ of size <N> in workgroup ID (1,2,3)
```

is

```bash
(gdb) break __asan_report_load<N>
```

It is possible to set breakpoints on all ASan report functions using these commands:

```bash
$ rocgdb <path to application>
(gdb) start <commmand line arguments>
(gdb) rbreak ^__asan_report
(gdb) c
```

## Using ASan with a short HIP application

Consider the following simple and short demo of using the Address Sanitizer with a HIP application:

```C++

#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 1;
}

int
main(int argc, char **argv)
{
    int m = std::atoi(argv[1]);
    int n1 = std::atoi(argv[2]);
    int n2 = std::atoi(argv[3]);
    int c = std::atoi(argv[4]);
    int *dp;
    hipMalloc(&dp, m*sizeof(int));
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
    int *hp = (int*)malloc(c * sizeof(int));
    hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    hipFree(dp);
    free(hp);
    std::puts("Done.");
    return 0;
}
```

This application will attempt to access invalid addresses for certain command line arguments. In particular, if `m < n1 * n2` some device threads will attempt to access
unallocated device memory.

Or, if `c < m`, the `hipMemcpy` function will copy past the end of the `malloc` allocated memory.

**Note**: The `hipcc` compiler is used here for simplicity.

Compiling without XNACK results in a warning.

```bash
$ hipcc -g --offload-arch=gfx90a:xnack- -fsanitize=address -shared-libsan mini.hip -o mini
clang++: warning: ignoring` `-fsanitize=address' option for offload arch 'gfx90a:xnack-`, as it is not currently supported there. Use it with an offload arch containing 'xnack+' instead [-Woption-ignored]`.
```

The binary compiled above will run, but the GPU code will not be instrumented and the `m < n1 * n2` error will not be detected. Switching to `--offload-arch=gfx90a:xnack+` in the command above results in a warning-free compilation and an instrumented application. After setting `PATH`, `LD_LIBRARY_PATH` and `HSA_XNACK` as described earlier, a check of the binary with `ldd` yields the following,

```bash
$ ldd mini
        linux-vdso.so.1 (0x00007ffd1a5ae000)
        libclang_rt.asan-x86_64.so => /opt/rocm-6.1.0-99999/llvm/lib/clang/17.0.0/lib/linux/libclang_rt.asan-x86_64.so (0x00007fb9c14b6000)
        libamdhip64.so.5 => /opt/rocm-6.1.0-99999/lib/asan/libamdhip64.so.5 (0x00007fb9bedd3000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fb9beba8000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fb9bea59000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fb9bea3e000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fb9be84a000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fb9be844000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fb9be821000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fb9be817000)
        libamd_comgr.so.2 => /opt/rocm-6.1.0-99999/lib/asan/libamd_comgr.so.2 (0x00007fb9b4382000)
        libhsa-runtime64.so.1 => /opt/rocm-6.1.0-99999/lib/asan/libhsa-runtime64.so.1 (0x00007fb9b3b00000)
        libnuma.so.1 => /lib/x86_64-linux-gnu/libnuma.so.1 (0x00007fb9b3af3000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fb9c2027000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fb9b3ad7000)
        libtinfo.so.6 => /lib/x86_64-linux-gnu/libtinfo.so.6 (0x00007fb9b3aa7000)
        libelf.so.1 => /lib/x86_64-linux-gnu/libelf.so.1 (0x00007fb9b3a89000)
        libdrm.so.2 => /opt/amdgpu/lib/x86_64-linux-gnu/libdrm.so.2 (0x00007fb9b3a70000)
        libdrm_amdgpu.so.1 => /opt/amdgpu/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1 (0x00007fb9b3a62000)

```

This confirms that the address sanitizer runtime is linked in, and the ASan instrumented version of the runtime libraries are used.
Checking the `PATH` yields

```bash
$ which llvm-symbolizer
/opt/rocm-6.1.0-99999/llvm/bin/llvm-symbolizer
```

Lastly, a check of the OS kernel version yields

```bash
$ uname -rv
5.15.0-73-generic #80~20.04.1-Ubuntu SMP Wed May 17 14:58:14 UTC 2023
```

which indicates that the required HMM support (kernel version > 5.6) is available. This completes the necessary setup. Running with `m = 100`, `n1 = 11`, `n2 = 10` and `c = 100` should produce
a report for an invalid access by the last 10 threads.

```bash
=================================================================
==3141==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc 0x7fb1410d2cc4
WRITE of size 4 in workgroup id (10,0,0)
  #0 0x7fb1410d2cc4 in set1(int*) at /home/dave/mini/mini.cpp:0:10

Thread ids and accessed addresses:
00 : 0x7fb14371d190 01 : 0x7fb14371d194 02 : 0x7fb14371d198 03 : 0x7fb14371d19c 04 : 0x7fb14371d1a0 05 : 0x7fb14371d1a4 06 : 0x7fb14371d1a8 07 : 0x7fb14371d1ac
08 : 0x7fb14371d1b0 09 : 0x7fb14371d1b4

0x7fb14371d190 is located 0 bytes after 400-byte region [0x7fb14371d000,0x7fb14371d190)
allocated by thread T0 here:
    #0 0x7fb151c76828 in hsa_amd_memory_pool_allocate /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors.cpp:692:3
    #1 ...

    #12 0x7fb14fb99ec4 in hipMalloc /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:568:3
    #13 0x226630 in hipError_t hipMalloc<int>(int**, unsigned long) /opt/rocm-6.1.0-99999/include/hip/hip_runtime_api.h:8367:12
    #14 0x226630 in main /home/dave/mini/mini.cpp:19:5
    #15 0x7fb14ef02082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16

Shadow bytes around the buggy address:
  0x7fb14371cf00: ...

=>0x7fb14371d180: 00 00[fa]fa fa fa fa fa fa fa fa fa fa fa fa fa
  0x7fb14371d200: ...

Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  ...
==3141==ABORTING
```

Running with `m = 100`, `n1 = 10`, `n2 = 10` and `c = 99` should produce a report for an invalid copy.

```shell
=================================================================
==2817==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x514000150dcc at pc 0x7f5509551aca bp 0x7ffc90a7ae50 sp 0x7ffc90a7a610
WRITE of size 400 at 0x514000150dcc thread T0
    #0 0x7f5509551ac9 in __asan_memcpy /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors_memintrinsics.cpp:61:3
    #1 ...

    #9 0x7f5507462a28 in hipMemcpy_common(void*, void const*, unsigned long, hipMemcpyKind, ihipStream_t*) /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:637:10
    #10 0x7f5507464205 in hipMemcpy /work/dave/git/compute/external/clr/hipamd/src/hip_memory.cpp:642:3
    #11 0x226844 in main /home/dave/mini/mini.cpp:22:5
    #12 0x7f55067c3082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16
    #13 0x22605d in _start (/home/dave/mini/mini+0x22605d)

0x514000150dcc is located 0 bytes after 396-byte region [0x514000150c40,0x514000150dcc)
allocated by thread T0 here:
    #0 0x7f5509553dcf in malloc /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_malloc_linux.cpp:69:3
    #1 0x226817 in main /home/dave/mini/mini.cpp:21:21
    #2 0x7f55067c3082 in __libc_start_main /build/glibc-SzIz7B/glibc-2.31/csu/../csu/libc-start.c:308:16

SUMMARY: AddressSanitizer: heap-buffer-overflow /work/dave/git/compute/external/llvm-project/compiler-rt/lib/asan/asan_interceptors_memintrinsics.cpp:61:3 in __asan_memcpy
Shadow bytes around the buggy address:
  0x514000150b00: ...

=>0x514000150d80: 00 00 00 00 00 00 00 00 00[04]fa fa fa fa fa fa
  0x514000150e00: ...

Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07
  Heap left redzone:       fa
  ...
==2817==ABORTING
```

## Known issues with using GPU sanitizer

* Red zones must have limited size. It is possible for an invalid access to completely miss a red zone and not be detected.

* Lack of detection or false reports can be caused by the runtime not properly maintaining red zone shadows.

* Lack of detection on the GPU might also be due to the implementation not instrumenting accesses to all GPU specific address spaces. For example, in the current implementation accesses to "private" or "stack" variables on the GPU are not instrumented, and accesses to HIP shared variables (also known as "local data store" or "LDS") are modeled in global memory and so are reported as global memory access issues.

* It can also be the case that a memory fault is reported for an invalid address even with the instrumentation. This is usually caused by the invalid address being so wild that its shadow address is outside any memory region, and the fault actually occurs on the access to the shadow address. It is also possible to hit a memory fault for the `NULL` pointer. While address 0 does have a shadow location, it is not poisoned by the runtime.
