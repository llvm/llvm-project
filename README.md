# AMD Fork of The LLVM Compiler Infrastructure

The AMD fork aims to contain all of [upstream LLVM](https://github.com/llvm/llvm-project), and also includes several AMD-specific additions in the `llvm-project/amd directory`:

- **amd/comgr** - The Code Object Manager API, designed to simplify linking, compiling, and inspecting code objects (code owner: [@lamb-j](https://www.github.com/lamb-j))
- **amd/device-libs** -The sources and CMake build system for a set of AMD-specific device-side language runtime libraries (code owner: [@b-sumner](https://www.github.com/b-sumner))
- **amd/hipcc** - A compiler driver utility that wraps clang and passes the appropriate include and library options for the target compiler and HIP infrastructure (code owner: [@david-salinas](https://www.github.com/david-salinas))

See the README files in respective subdirectories for more information on these AMD-specific projects. While the AMD fork aims to otherwise follow upstream as closely as possible, there are several outstanding differences.

- *OpenMP* - The AMD fork contains several changes:
    * Additional optimizations for OpenMP offload
    * Host-exec services for printing on-device and doing malloc/free from device
    * Improved support for OMPT, the OpenMP tools interface
    * Driver improvements for multi-image and Target ID features
    * OMPD support, implements OpenMP D interfaces.
    * ASAN support for OpenMP.
    * MI300A Unified Shared Memory support

- *Heterogeneous Debugging* - A prototype of debug-info supporting AMDGPU targets, affecting most parts of the compiler, is implemented as documented in `docs/AMDGPULLVMExtensionsForHeterogeneousDebugging.rst` but is an ongoing work-in-progress. Fundamental changes are expected as parts of the design are adapted for upstreaming.
- *Address Sanitizer* - Changes were added to `santizer_common` and `asan` libraries in `compiler-rt` to support AMD GPU address sanitizer error detection and reports.  These changes are intended to be upstreamed.  The instrumentation pass changes have already been upstreamed.
- *Reverted Patches* - For upstream patches that break internal testing, we may temporarily revert these patches until the testing issues are resolved. We maintain a list of reverted upstream patches in `llvm-project/revert_patches.txt`.
