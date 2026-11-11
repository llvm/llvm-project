(platform_support)=

# Platform Support

LLVM-libc's targets differ in how complete they are. This page describes the
status of each. {doc}`arch_support` lists the
architectures that are tested continuously, and {ref}`porting` describes what
a target needs in order to be added or kept.

## Used in production

These targets have, or are expected to soon have, users deploying LLVM-libc in
a production environment. They are given some amount of priority over the more
experimental targets.

| Operating system | Architectures |
| --- | --- |
| Linux | x86-64, AArch64, RISC-V |
| Baremetal | AArch64, RISC-V, 32-bit Arm |

New functionality should be enabled on all platforms when possible, but may be
rolled out to just these targets first as the primary development targets. See
{ref}`code_organization`.

## Other targets

The remaining targets are partially complete or experimental. Breakage on them
is still a bug, but contributors are not expected to be able to test them.

| Target | Status |
| --- | --- |
| GPU (AMDGPU, NVPTX) | Maintained by the GPU maintainers. See {doc}`the GPU docs <gpu/index>`. |
| Baremetal Hexagon | Newer target (as of 2026) but hopefully an area of growth. |
| FreeBSD | A port is in progress. It has a maintainer and presubmit CI. |
| macOS and Windows | Overlay mode only. The overlay tests run in presubmit, but both ports need significant work, and a full build for these systems has not been designed. |
| Linux on 32-bit Arm | Primarily used for testing on 32-bit Arm since baremetal CI is challenging. |
| Linux on PowerPC | Minimal. It exists mainly so that the code shared with libc++ keeps building there. |
| Linux on i386 | Theoretically [in the works](https://github.com/llvm/llvm-project/issues/93709), though it may be sunset. |
| UEFI | Experimental and incomplete. See {doc}`the UEFI docs <uefi/index>`. |

LLVM-libc is also being integrated into the Android and Fuchsia operating
systems via {doc}`overlay mode <overlay_mode>`.

## Breakages

A change that breaks any buildbot should be reverted or, if the fix is trivial,
fixed forward. This is intended to follow LLVM's
[patch reversion policy](https://llvm.org/docs/DeveloperPolicy.html#patch-reversion-policy).

## Operating system versions

For Linux, we support kernel versions as listed on
[kernel.org](https://kernel.org/), including `longterm` (not past EOL
date), `stable`, and `mainline` versions. We actively adopt new features
from `linux-next`.

For Windows, we plan to support products within their lifecycle. Please refer to
[Search Product and Services Lifecycle Information](https://learn.microsoft.com/en-us/lifecycle/products/?products=windows) for more information.

LLVM-libc does not guarantee backward compatibility with operating systems that
have reached their EOL. Compatibility patches for obsolete operating systems
will not be accepted.
