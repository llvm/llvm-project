# Return Address Authentication Hardening

```{contents}
:local:
```

## Introduction

Return Address Authentication Hardening is a mitigation against the
PACMAN attack, which aims to bypass Pointer Authentication on AArch64
targets. The hardening mechanism described here is specific to pointer
authentication of return addresses.

Return Address Signing (or Authentication), also known as pac-ret, is a
feature devised to protect programs against Return Oriented Programming
(ROP), in which attackers may hijack the return address of functions in
order to direct execution to malicious code.

pac-ret can be enabled via different command-line options:

- `-mbranch-protection=` with `pac-ret`, `pac-ret+leaf` or `standard` as
  value.
- `-msign-return-address=` with `non-leaf` or `all` as value.
- `-fptrauth-returns`.

More information can be found in
[Pointer Authentication](PointerAuthentication.md).

Return Address Authentication Hardening is a mechanism to strenghthen
Return Address Signing against the PACMAN attack in AArch64 targets. It
can be enabled with `-mharden-pac-ret=load-return-address`.

## PACMAN attack

PACMAN is an attack that aims to extract valuable information about
pointer authentication codes using side-channels in speculative
execution.

It is performed with the use of gadgets to try and guess PAC codes.
These guesses raise no faults because they are done in speculation. By
observing the effects of the guessed PAC code on the processor's cache,
it might be possible to determine the valid PAC code for the address to
which the attacker wants the program to return.

A usual PACMAN gadget looks like this:

```c
void function() {
  ...
  if (condition)
    return;
  ...
}
```

Such code would be compiled to:

```asm
paciasp
...
cbz w0, .LBB0_2
autiasp
ret
...
```

This code segment may be used as a gadget. A speculative execution of
this segment can happen as follows:

- If the Link Register (LR) has the right PAC code, `autiasp` will
  succeed and strip the PAC code out of it. The processor's instruction
  fetcher will then bring the code after the return into the cache (that
  is, the instructions located at the address pointed by LR).

- If the LR has the wrong PAC code, `autiasp` will not succeed and
  hence will write a predefined error value to the LR's higher bits.
  Because of this, the instruction fetcher will not bring the code
  after the return into the cache.

This difference in behavior is what drives the PACMAN attack. An
attacker can try to guess PAC codes and monitor cache behavior until the
code after the return is observed to have been brought into the cache.

Details can be found at [pacmanattack.com](https://pacmanattack.com).

## Hardening

In order to mitigate the PACMAN attack, a hardening mechanism can be
enabled with `-mharden-pac-ret=load-return-address`.

```asm
paciasp
...
cbz w0, .LBB0_2
autiasp
mov     x8, x30
xpaclri
ldr     w30, [x30]
ret     x8
...
```

The idea is to always bring the code after the return into cache (the
instructions located at the address pointed by LR), therefore minimizing
the difference between a speculative execution with a correct PAC code
and with an incorrect one.

- `autiasp` performs the authentication step.
- `mov x8, x30` copies the return address (LR and x30 are synonyms) to a
  temporary.
- `xpaclri` strips the PAC code out of the return address in x30.
- `ldr w30, [x30]` performs a load of the return address in x30.
- `ret x8` returns to the authenticated return address.

The load operation brings the code into the cache even if the
authentication step fails. As a consequence, in either case the code is
loaded into the cache. Furthermore, the return operation uses the
original return address before stripping, so the return address
protection is still kept in place in a normal non-speculative execution.

If FEAT_PAUTH is present, the code sequence can use instructions only
available with said feature with no change in semantics:

```asm
autiasp
mov     x8, x30
xpaci   x8
ldr     w8, [x8]
ret
```

## Command-line option

Return address authentication hardening can be enabled at translation
unit level with `-mharden-pac-ret=load-return-address`. It requires
pac-ret to be enabled at translation unit level as well.

## Function attribute

In addition to the command-line option
`-mharden-pac-ret=load-return-address`, the developer can enable the
mitigation at function level with the use of the corresponding function
attribute.

```c
__attribute__((target(
    "branch-protection=pac-ret,harden-pac-ret=load-return-address")))
void function() {
  ...
}
```

## Caveats

The load of return address brings the code into the shared
instruction/data cache, therefore this cache level can't be used as an
oracle to find out whether the authentication succeeded or not. However,
in the case of authentication success, the code is also fetched into the
instruction cache. An attacker who is able to measure this cache level
specifically may still be able to carry out the exploit.

Another caveat is if the code at the return address contains a load
operation within the speculation window. If this is the case, this load
will only execute speculatively if authentication succeeds, thus opening
the program up for exploitation despite the mitigation.
