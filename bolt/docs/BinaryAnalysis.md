# BOLT-based binary analysis

As part of post-link-time optimization, BOLT needs to perform a range of analyses
on binaries such as reconstructing control flow graphs, and more.

The `llvm-bolt-binary-analysis` tool enables running requested analyses
on binaries, and generating reports. It does this by building on top of the
analyses implemented in the BOLT libraries.

Contents
1. [Background and motivation](#background-and-motivation)
2. [Usage](#usage)
3. [Pointer Authentication validator](#pointer-authentication-validator)
4. [How to add your own analysis](#how-to-add-your-own-analysis)

## Background and motivation

### Security scanners

For the past 25 years, a large numbers of exploits have been built and used in
the wild to undermine computer security. The majority of these exploits abuse
memory vulnerabilities in programs, see evidence from
[Microsoft](https://youtu.be/PjbGojjnBZQ?t=836),
[Chromium](https://www.chromium.org/Home/chromium-security/memory-safety/) and
[Android](https://security.googleblog.com/2021/01/data-driven-security-hardening-in.html).

It is not surprising therefore, that a large number of mitigations have been
added to instruction sets and toolchains to make it harder to build an exploit
using a memory vulnerability. Examples are: stack canaries, stack clash,
pac-ret, shadow stacks, arm64e, and many more.

These mitigations guarantee a so-called "security property" on the binaries they
produce. For example, for stack canaries, the security property is roughly that
a canary is located on the stack between the set of saved registers and the set
of local variables. For pac-ret, it is roughly that either the return address is
never stored/retrieved to/from memory; or, there are no writes to the register
containing the return address between an instruction authenticating it and a
return instruction using it.

From time to time, however, a bug gets found in the implementation of such
mitigations in toolchains. Also, code that is manually written in assembler
requires the developer to ensure these security properties by hand.

In short, it is sometimes found that a few places in the binary code are not
protected as well as expected given the requested mitigations. Attackers could
make use of those places (sometimes called gadgets) to circumvent the protection
that the mitigation should give.

One of the reasons that such gadgets, or holes in the mitigation implementation,
exist is that typically the amount of testing and verification for these
security properties is limited to checking results on specific examples.

In comparison, for testing functional correctness, or for testing performance,
toolchain and software in general typically get tested with large test suites
and benchmarks. In contrast, this typically does not get done for testing the
security properties of binary code.

Unlike functional correctness where compilation errors result in test failures,
and performance where speed and size differences are measurable, broken security
properties cannot be easily observed using existing testing and benchmarking
tools.

The security scanners implemented in `llvm-bolt-binary-analysis` aim to enable
the testing of security hardening in arbitrary programs and not just specific
examples.

### Pointer Authentication

[Pointer Authentication](https://clang.llvm.org/docs/PointerAuthentication.html)
is intended to make it harder for an attacker to replace pointers at run time.
This is achieved by making it possible for the compiler or the programmer to
produce a *signed* pointer from a raw one, and then to probabilistically
*authenticate the signature* at another site in the program.
On AArch64 this is achieved by injecting a cryptographic hash, called a
["Pointer Authentication Code" (PAC)](https://llsoftsec.github.io/llsoftsecbook/#sec:pointer-authentication),
to the upper bits of the pointer.
While this approach can be applied to any pointers in the program, the most
frequent use case, at least in C and C++, is protecting the code pointers.
The language rules for such pointers are more restrictive, thus allowing the
compiler to implement various hardenings transparently to the programmer.

Probably the most simple variant of hardening based on Pointer Authentication is
[`pac-ret`](https://llsoftsec.github.io/llsoftsecbook/#sec:pac-ret), a security
hardening scheme implemented in compilers such as GCC and Clang, using the
command line option `-mbranch-protection=pac-ret`. This option is enabled by
default on most widely used Linux distributions. The hardening scheme mitigates
[Return-Oriented Programming (ROP)](https://llsoftsec.github.io/llsoftsecbook/#return-oriented-programming)
attacks by making sure that return addresses are only ever stored to memory
in a signed form. This makes it substantially harder for attackers to divert
control flow by overwriting a return address with a different value.

The approach to validation of Pointer Authentication hardening implemented in
`llvm-bolt-binary-analysis` is tracking register safety using dataflow analysis.
At each program point it is computed whether the particular register can be
controlled and whether it can be inspected by an attacker under
[Pointer Authentication threat model](https://clang.llvm.org/docs/PointerAuthentication.html#theory-of-operation).
Then, for a number of sensitive instruction kinds (such as function calls and
pointer signing instructions), the properties of input or output operands are
inspected to check if the particular instruction is emitted in a safe manner.

## Usage

```
llvm-bolt-binary-analysis --scanners=<list> [options] <binary>
```

The `--scanners=` option accepts a comma-separated list of analyses to run on
the provided binary. The binary to be analyzed can be either ELF executable or
shared object. Similar to other BOLT tools, `llvm-bolt-binary-analysis` expects
`binary` to be unstripped and preferably linked with `--emit-relocs` linker option.

In addition to options printed by `llvm-bolt-binary-analysis --help-hidden`,
other relevant BOLT options can generally be passed, see `llvm-bolt --help-hidden`.
Incomplete help message is a known issue and is tracked in
[#176969](https://github.com/llvm/llvm-project/issues/176969).

The only analysis which is currently implemented is validation of Pointer
Authentication hardening applied to the binary.
The specific set of gadget kinds which are searched for depends on command line
options. Each gadget found by PtrAuth gadget scanner results in a plain text
report printed at the end of the analysis.
Furthermore, an attempt is made to provide an extra information on the
instructions that made the register unsafe.
Please note that this extra information is provided on a best-effort basis and
is not expected to be as accurate as the reports themselves.

Here is an example of the report:

```
GS-PAUTH: signing oracle found in function function_name, basic block .LBB08, at address 102b8
  The instruction is     000102b8:      pacda   x0, x1
  The 1 instructions that write to the affected registers after any authentication are:
  1.     000102b4:      ldr     x0, [x1]
  This happens in the following basic block:
    000102b4:   ldr     x0, [x1]
    000102b8:   pacda   x0, x1
    000102bc:   ret
```

A similar report without the associated extra information looks like this:

```
GS-PAUTH: signing oracle found in function function_name, basic block .LBB016, at address 10384
  The instruction is     00010384:      pacda   x0, x1
  The 0 instructions that write to the affected registers after any authentication are:
```

## Pointer Authentication validator

Pointer Authentication analysis is able to search for a number of gadget kinds,
with the specific set depending on command line options:
* [`ptrauth-pac-ret`](#return-address-protection-ptrauth-pac-ret) -
  non-protected return instruction
* [`ptrauth-tail-calls`](#return-address-protection-before-tail-call-ptrauth-tail-calls) -
  performing a tail call with an untrusted value in the link register
* [`ptrauth-forward-cf`](#indirect-branch-call-target-protection-ptrauth-forward-cf) -
  non-protected destination of branch or call instruction
* [`ptrauth-sign-oracles`](#signing-oracles-ptrauth-sign-oracles) -
  signing of untrusted value (signing oracle)
* [`ptrauth-auth-oracles`](#authentication-oracles-ptrauth-auth-oracles) -
  revealing the result of authentication without crashing the program on failure
  (authentication oracle)

Validation is performed by `llvm-bolt-binary-analysis` on a per-function basis.
First, the register properties are computed by analyzing the function as a whole.
Then, the instructions are considered in isolation. For each kind of gadget,
the set of susceptible instructions is computed. The properties of input or
output registers of each such instruction are analyzed and reports are produced
for unsafe instruction usage.

Each gadget kind that is searched for can be characterized by the combination of
* the set of instructions to analyze
* the properties of input or output operands to check

Currently, three properties can be computed for each register at any given
program point:
* **"trusted"** - the register is known not to be attacker-controlled, either because
  it successfully passed authentication or because its value was materialized
  using an instruction sequence that an attacker cannot tamper with
  * **"safe-to-dereference"** (sometimes referred to as "s-t-d" below) -
    a weaker property that the register can be controlled by an attacker to some
    extent, but any memory access using a value crafted by an attacker is known
    to result in an access to an unmapped memory ("segmentation fault").
    This makes it possible for authentication instructions to return an invalid
    address on failure as long as it is known to crash the program on accessing
    memory, but may requires extra care to be taken when implementing operations
    like re-signing a pointer with a different signing schema without accessing
    that address in-between. If any failed authentication instruction is
    guaranteed to terminate the program abnormally, then "safe-to-dereference"
    and "trusted" properties are equivalent.
* **"cannot escape unchecked"** - at every possible execution path after this point,
  it is known to be impossible for an attacker to determine that the value is
  a result of a failed authentication operation (for example, the register is
  zeroed, or its value is checked to be valid, so that failure results in
  immediate abnormal program termination).

Generally, BOLT strives to reconstruct the control-flow graph of each function,
which is important for dataflow analysis. However, when BOLT fails to recognize
some control flow in the particular function, that function ends up being
represented as a flat list of instructions - in such cases
`llvm-bolt-binary-analysis` computes register properties using a fallback
analysis implementation, which is less precise.

The below sub-sections describe the particular detectors. Please note that while
the descriptions refer to AArch64 for simplicity, the implementation of gadget
detectors in `llvm-bolt-binary-analysis` attempts to be target-neutral by
isolating AArch64 specifics in target-dependent hooks.

### Return address protection (`ptrauth-pac-ret`)

**Instructions:** Return instructions without built-in authentication:
either `ret` (implicit `x30` register) or `ret <reg>`, but not `retaa` and
similar instructions.

**Property:** The register holding the return address must be safe-to-dereference.

**Notes:** Cross-exception-level return instructions (`eret`) are not analyzed yet.

A report is generated for a return instruction whose destination is possibly
attacker-controlled.

**Examples:**
```asm
authenticated_return:
  pacibsp
  ; ...
  ; ... some code here ...
  ; ...
  retab ; Built-in authentication, thus out of scope.

good_leaf_function:
  ; x30 is implicitly safe-to-dereference (s-t-d) and trusted at function entry.
  mov     x0, #42
  ; x30 was not written to by this function, thus remains s-t-d.
  ret

good_non_leaf_function:
  pacibsp

  ; Spilling signed return address.
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp

  bl      callee

  ; Re-loading signed return address.
  ; LDP writes to x30 and thus resets it to neither s-t-d nor trusted state.
  ldp     x29, x30, [sp], #16

  ; Checking that signature is valid.
  ; AUTIBSP sets "s-t-d" property of x30, but not "trusted" (unless FEAT_FPAC
  ; is known to be implemented).
  autibsp

  ; x30 is s-t-d at this point.
  ret

bad_spill:
  ; x30 is implicitly s-t-d at function entry.
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp

  bl      callee ; Spilled x30 may have been overwritten on stack.

  ; Writing to x30 resets its s-t-d property.
  ldp     x29, x30, [sp], #16
  ; x30 is unsafe by the time it is used by ret, thus generating a report.
  ret

bad_clobber:
  pacibsp
  ; ...
  ; ... some code here ...
  ; ...
  autibsp
  mov     x30, x1
  ; The value in LR is unsafe, even though there was autibsp above.
  ret
```

### Return address protection before tail call (`ptrauth-tail-calls`)

**Instructions:** Branch instructions (both direct and indirect, regular or
with built-in authentication), classified as tail calls either by BOLT or by
PtrAuth gadget scanner's heuristic.

**Property:** link register (`x30` on AArch64) must be trusted.

**Notes:** Heuristics are involved to classify instructions either as a tail
call or as another kind of branch (such as jump table or computed goto).

A report is generated if tail call is performed with untrusted link register.
This basically means that the tail-called function would have link register
untrusted on its entry (unlike inherently correct address placed to link
register by one of `bl*` instructions when non-tail call is performed).

```asm
non_protected_tail_call:
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp
  bl      callee
  ldp     x29, x30, [sp], #16
  ; x30 is neither trusted nor safe-to-dereference at this point.
  b       tail_callee

non_checked_tail_call:
  pacibsp
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp
  bl      callee
  ldp     x29, x30, [sp], #16
  autibsp
  ; x30 is safe-to-dereference, but not fully trusted at this point.
  b       tail_callee

tail_callee:
  pacibsp
  ; ...
```

Even though `x30` is likely to be safe-to-dereference before exit from a function
(whether via return or tail call) in a consistently pac-ret-protected program,
with respect to this gadget kind it further must be fully "trusted".
With `x30` being safe-to-dereference, but not fully trusted at the entry to the
tail callee, the subsequent `pacibsp` instruction may act as a [signing oracle](#signing-oracles-ptrauth-sign-oracles).

**FIXME:** Is it actually possible when none of `FEAT_FPAC`, `FEAT_EPAC`, or `FEAT_PAuth2` are implemented?

Properly mitigating this issue would usually require inserting an explicit
check after a regular authentication instruction, which may be either too
expensive (if a fully-generic XPAC-based sequence is being used) on one hand,
or not required at all (if `FEAT_FPAC` is known to be implemented) on the other hand.

### Indirect branch / call target protection (`ptrauth-forward-cf`)

**Instructions:** Indirect call and branch instructions without built-in
authentication: either `blr <reg>` or `br <reg>`, but not `blraa`, `braa`
and similar instructions.

**Property:** Call or branch target register must be safe-to-dereference.

Report is generated for an indirect branch or call instruction whose destination
is possibly attacker-controlled.

**Examples:**

```asm
direct_call:
  ; ...
  bl     callee ; Direct call, thus out of scope.
  ; ...

authenticated_call:
  ; ...
  ldr     x2, [x1]
  blraa   x2, x1   ; Built-in authentication, thus out of scope.
  ; ...

good_call:
  ; ...
  ldr     x2, [x1]
  autia   x2, x1
  blr     x2
  ; ...

bad_call:
  ; ...
  ldr     x2, [x1]
  autia   x2, x1
  ; Store unprotected address.
  stp     x2, [x3]
  ; ...
  ; The callee address may have been overwritten in memory.
  ldr     x2, [x3]
  blr     x2
  ; ...

good_call_dataflow:
  cbz     x0, .L1
  ldr     x2, [x1]
  autia   x2, x1
  b       .L2
.L1:
  adrp    x2, callee
  add     x2, x2, :lo12:callee
.L2:
  ; Dataflow analysis can deduce that x2 is s-t-d on any possible execution
  ; path leading to the below "br x2" instruction.
  br      x2

bad_call_dataflow:
  cbz     x0, .L3
  adrp    x2, callee
  add     x2, x2, :lo12:callee
.L3:
  ; x2 is untrusted is x0 is 0.
  br      x2

```

### Signing oracles (`ptrauth-sign-oracles`)

**Instructions:** Address-signing instructions.

**Property:** The address being signed must be trusted.

Reports signing of untrusted values, as this could make arbitrary and possibly
attacker-controlled values indistinguishable from perfectly trusted and protected ones.

**FIXME:** Is `aut** + pac**` sequence actually exploitable when none of
`FEAT_FPAC`, `FEAT_EPAC`, or `FEAT_PAuth2` are implemented?

**Examples:**

```asm
good_sign_constant:
  ; ...
  adrp    x0, sym
  add     x0, x0, :lo12:sym
  pacda   x0, x1
  ; ...

good_resign:
  ; ...
  autda   x0, x1
  ; x0 is s-t-d here.
  ldr     x2, [x0]
  ; If we got here without crashing on the above ldr, x0 is fully trusted.
  pacdb   x0, x1
  ; ...

bad_resign_if_not_fpac:
  ; ...
  autda   x0, x1
  ; x0 is only s-t-d, but not trusted here, unless autda raises an error on failure.
  pacdb   x0, x1
  ; ...

very_bad_function:
  pacda   x0, x1
  ret
```

### Authentication oracles (`ptrauth-auth-oracles`)

**Instructions:** Standalone authentication instructions: `autda`, `autdb`, etc.
(i.e. not those built into corresponding memory-accessing instructions, such as
`ldraa` or `blraa`).

**Property:** The **result** of authentication must be written to a register
that cannot escape unchecked.

Reports authentication instructions, whose result (success or failure) can be
observed by the attacker and used to guess the correct PAC field by trial-and-error.

The authentication oracles searched for by this detector are impossible if all
authentication instructions are known to generate an irrecoverable error on
failure. On AArch64 this is the case if CPU is known to implement `FEAT_FPAC`
(though recoverability depends on the OS: for example, on Linux such errors
result in signals that can be handled if configured accordingly).
This check is disabled by `--auth-traps-on-failure` command line option.

**Examples:**

```asm
; The descriptions assume FEAT_FPAC is not implemented.

good_auth_call:
  paciasp
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp

  ; The result of authentication is inevitably dereferenced by blr.
  ; If autia returns a non-canonical address due to incorrect signature,
  ; the program crashes when the resulting address is jumped-to by blr.
  cbz     x2, .L1
  autia   x0, x1
  blr     x0

.L1:
  ldp     x29, x30, [sp], #16
  autiasp
  ret

bad_auth_call:
  paciasp
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp

  ; x0 may be observed by the caller of bad_auth_call if x2 is 0.
  autia   x0, x1
  cbz     x2, .L2
  blr     x0

.L2:
  ldp     x29, x30, [sp], #16
  autiasp
  ret

bad_leaks_to_callee:
  paciasp
  stp     x29, x30, [sp, #-16]!
  mov     x29, sp

  ldr     x20, [x0]
  autda   x20, x0
  ; The result of authentication is leaked to the called function.
  bl      callee
  ; The below ldr instruction would properly check x20 if placed above the call.
  ldr     x0, [x20]

  ldp     x29, x30, [sp], #16
  autiasp
  ret
```

### Known issues and missing features

#### Control-flow graph availability

The analysis quality is degraded if BOLT is unable to reconstruct control-flow
graph of the function correctly.

When reconstructing CFG for a particular function, it is possible that BOLT
finds a code pattern that it is unable to handle. In that case PtrAuth validator
processes a function containing a flat list of instructions (as opposed to
interlinked basic blocks) and the reports (if any) are printed without the
`, basic block <name>` part in the first line.

Furthermore, it is possible that CFG information is returned by BOLT for a
function, even though the graph is imprecise (known issues are tracked in
[#177761](https://github.com/llvm/llvm-project/issues/177761),
[#178058](https://github.com/llvm/llvm-project/issues/178058),
[#178232](https://github.com/llvm/llvm-project/issues/178232)).
Inaccurate CFG information may result in false positives and false negatives,
thus PtrAuth gadget scanner produces warning messages (at most once per
function) for non-entry basic blocks without any predecessors in CFG: while
unreachable basic blocks are technically correct, truly unreachable blocks are
unlikely to exist in optimized code.

#### Last writing instructions

Some of the reports contain extra information along these lines

```
  The 1 instructions that write to the affected registers after any authentication are:
  1.     000102b4:      ldr     x0, [x1]
  This happens in the following basic block:
    000102b4:   ldr     x0, [x1]
    000102b8:   pacda   x0, x1
    000102bc:   ret
```

This information is provided on a best-effort basis and is less reliable than
gadget reports themselves.

#### Feature: scan for unsafe computation of discriminator value

There is a common pattern on AArch64 to compute the discriminator as a blend
of an address and an integer modifier by inserting a compile-time constant
value into the top 16 bits of the storage address. It is not necessarily
possible to prevent an attacker from modifying the storage address, but the
insertion of 16-bit constant modifier can always be performed immediately before
the discriminator value is used by signing or authentication instruction.

While not as bad as signing an arbitrary value or spilling an already authenticated
value to memory, spilling a "ready-to-use" discriminator value instead of computing
it right before usage is something we would rather avoid. On the other hand,
using an arbitrary value as the discriminator should probably be allowed, making
it hard to distinguish the below patterns:

```asm
; Valid and not reported.
good_store_with_address_and_constant_diversity:
  mov     x16, x1
  movk    x16, #1234, lsl #48
  pacda   x0, x16
  str     x0, [x1]
  ret

; Valid and not reported.
good_store_with_address_diversity:
  pacda   x0, x1
  str     x0, [x1]
  ret

; Spilled discriminator. Not critically wrong, but could rather be avoided.
; Not reported, but probably should (false negative).
bad_spilling:
  mov     x16, x1
  movk    x16, #1234, lsl #48

  ; Spilling the discriminator to memory.
  str     x16, [x2]
  ; Reloaded value could have been modified by an attacker.
  ldr     x16, [x2]

  pacda   x0, x16
  str     x0, [x1]
  ret

; Not reported (and should probably not).
better_spilling:
  mov     x16, x1
  str     x16, [x2]
  ; Reloaded value could have been modified by an attacker.
  ldr     x16, [x2]

  movk    x16, #1234, lsl #48
  pacda   x0, x16
  str     x0, [x1]
  ret
```

#### Handling of constants

While (PC-relative) address constants are tracked as "trusted" register state
by `SrcSafetyAnalysis`, constant values are not generally accounted for.

This results in false-positives like reporting

```asm
  ; Let assume FEAT_FPAC is implemented.
  autda   x16, x22
  mov     x17, #0x128
  add     x16, x16, x17
  pacda   x16, x22
```

as a signing oracle, even though

```asm
  ; Let assume FEAT_FPAC is implemented.
  autda   x16, x22
  add     x16, x16, #8
  pacda   x16, x22
```

is not reported, because `add Xdst, Xsrc, #imm` is recognized as a safe address
computation.

As an example of false-negative, it is possible that an instruction like
`add x0, x0, #1` could be called in a loop with an attacker-controlled number
of iterations, making it technically possible for an attacker to replace a
valid pointer with a pointer to an arbitrary *higher* address.

#### Tail call detection

`ptrauth-tail-calls` detector uses a heuristic to classify each branch
instruction either as a tail call or as an unrelated branch instruction.

Most other parts of BOLT should not break code when rewriting. Unlike them,
this analyzer tries to keep reasonable balance between false positive reports
and missed issues. For this reason, it inspects some other branch instructions
in addition to those BOLT is sure about.

While it should provide reasonable balance between false positives and false
negatives on general code, this may not be the case for some specific patterns.
An example of a generally uncommon code pattern that is likely to yield false
positives is [labels-as-values GCC extension](https://gcc.gnu.org/onlinedocs/gcc/Labels-as-Values.html)
which is also supported by Clang.

#### Other known issues

* Not handling "no-return" functions. See issue
  [#115154](https://github.com/llvm/llvm-project/issues/115154) for details and
  pointers to open PRs to fix this.
* Scanning of binaries compiled by Clang at `-Oz` optimization level produces a
  lot of reports due to outlining. Many such reports could probably be considered
  false negatives as long as an attacker is unable to call `OUTLINED_FUNCTION`s
  as ROP gadgets in the first place.
* False positives are possible due to multi-instruction pointer-checking sequences
  not being detected without CFG.
* While obviously "checking" the result of pointer authentication, store
  instructions do not transition their address operand register from safe-to-
  dereference to trusted state yet. This does not affect scanning regular code
  hardened neither by `pac-ret`, nor by `arm64e` or `pauthtest`.

## How to add your own analysis

### Pointer Authentication validator

To implement the detection of a new gadget kind, add new
`shouldReport*Gadget` function to `bolt/lib/Passes/PAuthGadgetScanner.cpp` and
call it either from `FunctionAnalysisContext::findUnsafeUses` or
`FunctionAnalysisContext::findUnsafeDefs`.

To improve overall analysis quality by better computing register properties,
either modify one of `*SafetyAnalysis` classes in `PAuthGadgetScanner.cpp`
(if the improvement is target-neutral), or one of target-specific hooks in
the subclass of `MCPlusBuilder` corresponding to your target (if an analysis of
target-specific instruction patterns is to be improved).

To add support for a new target, if one eventually implements similar pointer
protection technique, implement PtrAuth-related hooks in the subclass of
`MCPlusBuilder` corresponding to your target. Ideally, no changes should be
needed in `PAuthGadgetScanner.cpp`, as it is intended to be reasonably target-
independent, though it is possible that some amount of further generalization
may be required.

### New types of analyses

_TODO: this section needs to be written. Ideally, we should have a simple
"example" or "template" analysis that can be the starting point for implementing
custom analyses_
