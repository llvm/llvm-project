# BOLT-based binary analysis

As part of post-link-time optimizing, BOLT needs to perform a range of analyses
on binaries such as reconstructing control flow graphs, and more.

The `llvm-bolt-binary-analysis` tool enables running requested binary analyses
on binaries, and generating reports. It does this by building on top of the
analyses implemented in the BOLT libraries.

## Which binary analyses are implemented?

* [Security scanners](#security-scanners)
  * [pac-ret analysis](#pac-ret-analysis)

### Security scanners

For the past 25 years, a large numbers of exploits have been built and used in
the wild to undermine computer security. The majority of these exploits abuse
memory vulnerabilities in programs, see evidence from
[Microsoft](https://youtu.be/PjbGojjnBZQ?si=oCHCa0SHgaSNr6Gr&t=836),
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
mitigations in toolchains. Also, code that is written in assembler by hand
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


#### pac-ret analysis

`pac-ret` protection is a security hardening scheme implemented in compilers
such as GCC and Clang, using the command line option
`-mbranch-protection=pac-ret`. This option is enabled by default on most widely
used Linux distributions.

The hardening scheme mitigates
[Return-Oriented Programming (ROP)](https://llsoftsec.github.io/llsoftsecbook/#return-oriented-programming)
attacks by making sure that return addresses are only ever stored to memory with
a cryptographic hash, called a
["Pointer Authentication Code" (PAC)](https://llsoftsec.github.io/llsoftsecbook/#pointer-authentication),
in the upper bits of the pointer. This makes it substantially harder for
attackers to divert control flow by overwriting a return address with a
different value.

The hardening scheme relies on compilers producing appropriate code sequences when
processing return addresses, especially when these are stored to and retrieved
from memory.

The `pac-ret` binary analysis can be invoked using the command line option
`--scanners=pac-ret`. It makes `llvm-bolt-binary-analysis` scan through the
provided binary, checking each function for the following security property:

> For each procedure and exception return instruction, the destination register
> must have one of the following properties:
>
> 1. be immutable within the function, or
> 2. the last write to the register must be by an authenticating instruction. This
>    includes combined authentication and return instructions such as `RETAA`.

##### Example 1

For example, a typical non-pac-ret-protected function looks as follows:

```
        stp     x29, x30, [sp, #-0x10]!
        mov     x29, sp
        bl      g@PLT
        add     x0, x0, #0x3
        ldp     x29, x30, [sp], #0x10
        ret
```

The return instruction `ret` implicitly uses register `x30` as the address to
return to. Register `x30` was last written by instruction `ldp`, which is not an
authenticating instruction. `llvm-bolt-binary-analysis --scanners=pac-ret` will
report this as follows:

```
GS-PACRET: non-protected ret found in function f1, basic block .LBB00, at address 10310
  The return instruction is     00010310:       ret # pacret-gadget: pac-ret-gadget<Ret:MCInstBBRef<BB:.LBB00:6>, Overwriting:[MCInstBBRef<BB:.LBB00:5> ]>
  The 1 instructions that write to the return register after any authentication are:
  1.     0001030c:      ldp     x29, x30, [sp], #0x10
  This happens in the following basic block:
    000102fc:   stp     x29, x30, [sp, #-0x10]!
    00010300:   mov     x29, sp
    00010304:   bl      g@PLT
    00010308:   add     x0, x0, #0x3
    0001030c:   ldp     x29, x30, [sp], #0x10
    00010310:   ret # pacret-gadget: pac-ret-gadget<Ret:MCInstBBRef<BB:.LBB00:6>, Overwriting:[MCInstBBRef<BB:.LBB00:5> ]>
```

The exact format of how `llvm-bolt-binary-analysis` reports this is expected to
evolve over time.

##### Example 2: multiple "last-overwriting" instructions

A simple example that shows how there can be a set of "last overwriting"
instructions of a register follows:

```
        paciasp
        stp     x29, x30, [sp, #-0x10]!
        ldp     x29, x30, [sp], #0x10
        cbnz    x0, 1f
        autiasp
1:
        ret
```

This will produce the following diagnostic:

```
GS-PACRET: non-protected ret found in function f_crossbb1, basic block .Ltmp0, at address 102dc
  The return instruction is     000102dc:       ret # pacret-gadget: pac-ret-gadget<Ret:MCInstBBRef<BB:.Ltmp0:0>, Overwriting:[MCInstBBRef<BB:.LFT0:0> MCInstBBRef<BB:.LBB00:2> ]>
  The 2 instructions that write to the return register after any authentication are:
  1.     000102d0:      ldp     x29, x30, [sp], #0x10
  2.     000102d8:      autiasp
```

(Yes, this diagnostic could be improved because the second "overwriting"
instruction, `autiasp`, is an authenticating instruction...)

##### Known false positives or negatives

The following are current known cases of false positives:

1. Not handling "no-return" functions. See issue
   [#115154](https://github.com/llvm/llvm-project/issues/115154) for details and
   pointers to open PRs to fix this.
2. Not recognizing that a move of a properly authenticated value between registers,
   results in the destination register having a properly authenticated value.
   For example, the scanner currently produces a false negative for the following
   code sequence:
   ```
        autiasp
        mov     x16, x30
        ret     x16
   ```

The following are current known cases of false negatives:

1. Not handling functions for which the CFG cannot be reconstructed by BOLT. The
   plan is to implement support for this, picking up the implementation from the
   [prototype branch](
   https://github.com/llvm/llvm-project/compare/main...kbeyls:llvm-project:bolt-gadget-scanner-prototype).

## How to add your own binary analysis

_TODO: this section needs to be written. Ideally, we should have a simple
"example" or "template" analysis that can be the starting point for implementing
custom analyses_
