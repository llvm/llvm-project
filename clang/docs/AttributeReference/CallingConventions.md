## Calling Conventions

Clang supports several different calling conventions, depending on the target
platform and architecture. The calling convention used for a function determines
how parameters are passed, how results are returned to the caller, and other
low-level details of calling a function.

### aarch64_sve_pcs

{clang-attr-syntaxes}`AArch64SVEPcsDocs`

On AArch64 targets, this attribute changes the calling convention of a
function to preserve additional Scalable Vector registers and Scalable
Predicate registers relative to the default calling convention used for
AArch64.

This means it is more efficient to call such functions from code that performs
extensive scalable vector and scalable predicate calculations, because fewer
live SVE registers need to be saved. This property makes it well-suited for SVE
math library functions, which are typically leaf functions that require a small
number of registers.

However, using this attribute also means that it is more expensive to call
a function that adheres to the default calling convention from within such
a function. Therefore, it is recommended that this attribute is only used
for leaf functions.

For more information, see the documentation for `aarch64_sve_pcs` in the
ARM C Language Extension (ACLE) documentation.

[aarch64_sve_pcs]: https://github.com/ARM-software/acle/blob/main/main/acle.md#scalable-vector-extension-procedure-call-standard-attribute


### aarch64_vector_pcs

{clang-attr-syntaxes}`AArch64VectorPcsDocs`

On AArch64 targets, this attribute changes the calling convention of a
function to preserve additional floating-point and Advanced SIMD registers
relative to the default calling convention used for AArch64.

This means it is more efficient to call such functions from code that performs
extensive floating-point and vector calculations, because fewer live SIMD and FP
registers need to be saved. This property makes it well-suited for e.g.
floating-point or vector math library functions, which are typically leaf
functions that require a small number of registers.

However, using this attribute also means that it is more expensive to call
a function that adheres to the default calling convention from within such
a function. Therefore, it is recommended that this attribute is only used
for leaf functions.

For more information, see the documentation for [aarch64_vector_pcs][aarch64_vector_pcs] on
the Arm Developer website.

[aarch64_vector_pcs]: https://developer.arm.com/products/software-development-tools/hpc/arm-compiler-for-hpc/vector-function-abi


### fastcall

{clang-attr-syntaxes}`FastCallDocs`

On 32-bit x86 targets, this attribute changes the calling convention of a
function to use ECX and EDX as register parameters and clear parameters off of
the stack on return. This convention does not support variadic calls or
unprototyped functions in C, and has no effect on x86_64 targets. This calling
convention is supported primarily for compatibility with existing code. Users
seeking register parameters should use the `regparm` attribute, which does
not require callee-cleanup. See the documentation for [`__fastcall`][__fastcall] on MSDN.

[__fastcall]: http://msdn.microsoft.com/en-us/library/6xa169sk.aspx


### m68k_rtd

{clang-attr-syntaxes}`M68kRTDDocs`

On M68k targets, this attribute changes the calling convention of a function
to clear parameters off the stack on return. In other words, callee is
responsible for cleaning out the stack space allocated for incoming paramters.
This convention does not support variadic calls or unprototyped functions in C.
When targeting M68010 or newer CPUs, this calling convention is implemented
using the `rtd` instruction.


### ms_abi

{clang-attr-syntaxes}`MSABIDocs`

On non-Windows x86_64 and aarch64 targets, this attribute changes the calling convention of
a function to match the default convention used on Windows. This
attribute has no effect on Windows targets or non-x86_64, non-aarch64 targets.


### pcs

{clang-attr-syntaxes}`PcsDocs`

On ARM targets, this attribute can be used to select calling conventions
similar to `stdcall` on x86. Valid parameter values are "aapcs" and
"aapcs-vfp".


### preserve_all

{clang-attr-syntaxes}`PreserveAllDocs`

On X86-64 and AArch64 targets, this attribute changes the calling convention of
a function. The `preserve_all` calling convention attempts to make the code
in the caller even less intrusive than the `preserve_most` calling convention.
This calling convention also behaves identical to the `C` calling convention
on how arguments and return values are passed, but it uses a different set of
caller/callee-saved registers. This removes the burden of saving and
recovering a large register set before and after the call in the caller. If
the arguments are passed in callee-saved registers, then they will be
preserved by the callee across the call. This doesn't apply for values
returned in callee-saved registers.

- On X86-64 the callee preserves all general purpose registers, except for
  R11. R11 can be used as a scratch register. Furthermore it also preserves
  all floating-point registers (XMMs/YMMs).
- On AArch64 the callee preserve all general purpose registers, except X0-X8 and
  X16-X18. Furthermore it also preserves lower 128 bits of V8-V31 SIMD - floating
  point registers.

The idea behind this convention is to support calls to runtime functions
that don't need to call out to any other functions.

This calling convention, like the `preserve_most` calling convention, will be
used by a future version of the Objective-C runtime and should be considered
experimental at this time.


### preserve_most

{clang-attr-syntaxes}`PreserveMostDocs`

On X86-64 and AArch64 targets, this attribute changes the calling convention of
a function. The `preserve_most` calling convention attempts to make the code
in the caller as unintrusive as possible. This convention behaves identically
to the `C` calling convention on how arguments and return values are passed,
but it uses a different set of caller/callee-saved registers. This alleviates
the burden of saving and recovering a large register set before and after the
call in the caller. If the arguments are passed in callee-saved registers,
then they will be preserved by the callee across the call. This doesn't
apply for values returned in callee-saved registers.

- On X86-64 the callee preserves all general purpose registers, except for
  R11. R11 can be used as a scratch register. Floating-point registers
  (XMMs/YMMs) are not preserved and need to be saved by the caller.
- On AArch64 the callee preserve all general purpose registers, except X0-X8 and
  X16-X18.

The idea behind this convention is to support calls to runtime functions
that have a hot path and a cold path. The hot path is usually a small piece
of code that doesn't use many registers. The cold path might need to call out to
another function and therefore only needs to preserve the caller-saved
registers, which haven't already been saved by the caller. The
`preserve_most` calling convention is very similar to the `cold` calling
convention in terms of caller/callee-saved registers, but they are used for
different types of function calls. `coldcc` is for function calls that are
rarely executed, whereas `preserve_most` function calls are intended to be
on the hot path and definitely executed a lot. Furthermore `preserve_most`
doesn't prevent the inliner from inlining the function call.

This calling convention will be used by a future version of the Objective-C
runtime and should therefore still be considered experimental at this time.
Although this convention was created to optimize certain runtime calls to
the Objective-C runtime, it is not limited to this runtime and might be used
by other runtimes in the future too. The current implementation only
supports X86-64 and AArch64, but the intention is to support more architectures
in the future.


### preserve_none

{clang-attr-syntaxes}`PreserveNoneDocs`

On X86-64 and AArch64 targets, this attribute changes the calling convention of a function.
The `preserve_none` calling convention tries to preserve as few general
registers as possible. So all general registers are caller saved registers. It
also uses more general registers to pass arguments. This attribute doesn't
impact floating-point registers. `preserve_none`'s ABI is still unstable, and
may be changed in the future.

- On X86-64, only RSP and RBP are preserved by the callee.
  Registers R12, R13, R14, R15, RDI, RSI, RDX, RCX, R8, R9, R11, and RAX now can
  be used to pass function arguments. Floating-point registers (XMMs/YMMs) still
  follow the C calling convention.
- On AArch64, only LR and FP are preserved by the callee.
  Registers X20-X28, X0-X7, and X9-X14 are used to pass function arguments.
  X8, X16-X19, SIMD and floating-point registers follow the AAPCS calling
  convention. X15 is not available for argument passing on Windows, but is
  used to pass arguments on other platforms.


### regcall

{clang-attr-syntaxes}`RegCallDocs`

On x86 targets, this attribute changes the calling convention to
[`__regcall`][__regcall] convention. This convention aims to pass as many arguments
as possible in registers. It also tries to utilize registers for the
return value whenever it is possible.

[__regcall]: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/c-c-sycl-calling-conventions.html


### regparm

{clang-attr-syntaxes}`RegparmDocs`

On 32-bit x86 targets, the regparm attribute causes the compiler to pass
the first three integer parameters in EAX, EDX, and ECX instead of on the
stack. This attribute has no effect on variadic functions, and all parameters
are passed via the stack as normal.


### riscv::vector_cc, riscv_vector_cc, clang::riscv_vector_cc

{clang-attr-syntaxes}`RISCVVectorCCDocs`

The `riscv_vector_cc` attribute can be applied to a function. It preserves 15
registers namely, v1-v7 and v24-v31 as callee-saved. Callers thus don't need
to save these registers before function calls, and callees only need to save
them if they use them.


### riscv::vls_cc, riscv_vls_cc, clang::riscv_vls_cc

{clang-attr-syntaxes}`RISCVVLSCCDocs`

The `riscv_vls_cc` attribute can be applied to a function. Functions
declared with this attribute will utilize the standard fixed-length vector
calling convention variant instead of the default calling convention defined by
the ABI. This variant aims to pass fixed-length vectors via vector registers,
if possible, rather than through general-purpose registers.


### stdcall

{clang-attr-syntaxes}`StdCallDocs`

On 32-bit x86 targets, this attribute changes the calling convention of a
function to clear parameters off of the stack on return. This convention does
not support variadic calls or unprototyped functions in C, and has no effect on
x86_64 targets. This calling convention is used widely by the Windows API and
COM applications. See the documentation for [`__stdcall`][__stdcall] on MSDN.

[__stdcall]: http://msdn.microsoft.com/en-us/library/zxk0tw93.aspx


### sysv_abi

{clang-attr-syntaxes}`SysVABIDocs`

On Windows x86_64 targets, this attribute changes the calling convention of a
function to match the default convention used on Sys V targets such as Linux,
Mac, and BSD. This attribute has no effect on other targets.


### thiscall

{clang-attr-syntaxes}`ThisCallDocs`

On 32-bit x86 targets, this attribute changes the calling convention of a
function to use ECX for the first parameter (typically the implicit `this`
parameter of C++ methods) and clear parameters off of the stack on return. This
convention does not support variadic calls or unprototyped functions in C, and
has no effect on x86_64 targets. See the documentation for [`__thiscall`][__thiscall] on
MSDN.

[__thiscall]: http://msdn.microsoft.com/en-us/library/ek8tkfbw.aspx


### vectorcall

{clang-attr-syntaxes}`VectorCallDocs`

On 32-bit x86 *and* x86_64 targets, this attribute changes the calling
convention of a function to pass vector parameters in SSE registers.

On 32-bit x86 targets, this calling convention is similar to `__fastcall`.
The first two integer parameters are passed in ECX and EDX. Subsequent integer
parameters are passed in memory, and callee clears the stack. On x86_64
targets, the callee does *not* clear the stack, and integer parameters are
passed in RCX, RDX, R8, and R9 as is done for the default Windows x64 calling
convention.

On both 32-bit x86 and x86_64 targets, vector and floating point arguments are
passed in XMM0-XMM5. Homogeneous vector aggregates of up to four elements are
passed in sequential SSE registers if enough are available. If AVX is enabled,
256 bit vectors are passed in YMM0-YMM5. Any vector or aggregate type that
cannot be passed in registers for any reason is passed by reference, which
allows the caller to align the parameter memory.

See the documentation for [`__vectorcall`][__vectorcall] on MSDN for more details.

[__vectorcall]: http://msdn.microsoft.com/en-us/library/dn375768.aspx


