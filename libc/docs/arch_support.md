# Architecture Support

The table below lists what is built and tested on every pull request.

| Architecture | Tests run | Built only |
| --- | --- | --- |
| x86-64 | Linux (overlay and full build), Windows (overlay), FreeBSD (overlay) | UEFI |
| AArch64 | Linux (overlay and full build), macOS (overlay) | |
| 32-bit Arm | Linux (overlay, under QEMU) | Baremetal |
| 64-bit RISC-V | Linux (overlay, under QEMU) | |
| 32-bit RISC-V | | Baremetal |
| Hexagon | Baremetal | |
| AMDGPU | | GPU |

NVPTX is also a supported GPU target. See {doc}`the GPU docs <gpu/index>` for
both GPU targets.

After a change lands, the
[postcommit buildbots](https://lab.llvm.org/buildbot/#/builders?tags=libc)
currently cover Linux on x86-64: Clang and GCC, overlay and full build, and
AddressSanitizer.

i386 support is [in the works](https://github.com/llvm/llvm-project/issues/93709),
though it may be sunset. See {ref}`platform_support` for the status of each
target.

See {doc}`Bringup on a New OS or Architecture <porting>` for more
information. Please do first file a bug in
[our issue tracker](https://github.com/llvm/llvm-project/labels/libc) before
starting a port that you plan to upstream.
