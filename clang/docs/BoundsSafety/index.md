# `-fbounds-safety`

:::{warning}
Upstreaming of `-fbounds-safety` is still in progress, and the feature is not
yet available for users.
:::

`-fbounds-safety` is a C extension to enforce bounds safety to prevent
out-of-bounds (OOB) memory accesses, which remain a major source of security
vulnerabilities in C. `-fbounds-safety` aims to eliminate this class of bugs by
turning OOB accesses into deterministic traps.

## `-fbounds-safety` Documents

```{toctree}
:maxdepth: 1
:numbered:

BoundsSafety
BoundsSafetyAdoptionGuide
BoundsSafetyImplPlans
```

(bounds-safety-upstream-status)=
## Upstream Status

`-fbounds-safety` is being upstreamed incrementally: first the bounds
annotations, so that projects can start annotating their code, then the full
model. Until the model is complete, it is guarded by the experimental `-cc1`
flag `-fexperimental-bounds-safety`. A preview implementation is available in
the
[swiftlang/llvm-project](https://github.com/swiftlang/llvm-project/tree/stable/21.x)
fork.

`__counted_by`, `__sized_by` and their `_or_null` variants are available on
struct fields, with limits: the count must be a single identifier, and the
annotations are handled as declaration attributes rather than type attributes.
They are used by `-fsanitize=array-bounds` and `_FORTIFY_SOURCE=3`. Work is in
progress to handle them as type attributes, with late parsing, and to support
them on function parameters. The remaining annotations (`__single`,
`__ended_by`, `__null_terminated`, `__unsafe_indexable`) and complex count
expressions are planned. We aim to have the bounds annotations ready for
open-source projects to adopt by early 2027.

The full `-fbounds-safety` model, including wide pointers with bounds checks
and implicit bounds annotations, follows. Parts of it need RFCs.

The detailed status, with links to the relevant PRs and issues, is tracked in
[#238453](https://github.com/llvm/llvm-project/issues/238453).

## Open-source Adoption Effort

`-fbounds-safety` is currently available in the
[swiftlang/llvm-project](https://github.com/swiftlang/llvm-project/tree/stable/21.x)
fork, and has been adopted in millions of lines of security-critical code at
Apple, including the XNU networking stack, image and audio codecs, firmware,
and libraries used during secure boot.

Because `-fbounds-safety` is not yet available in mainline Clang, no external
project ships binaries built with it. However, some open-source projects are
adopting the model using the swiftlang fork: they make the required source
changes and add bounds annotations, without enabling `-fbounds-safety` in the
binaries they ship.

- [FreeBSD](https://wiki.freebsd.org/BoundsSafety)
- [libwebp](https://issues.webmproject.org/issues/432511225)

## How to Contribute

Contributions are welcome. Please coordinate with us before starting, by email
to <yeoul_na@apple.com> or in the
[#fbounds-safety](https://discord.com/channels/636084430946959380/1349963091432968287)
channel on the LLVM Discord. Open issues are tagged
[`clang:bounds-safety`](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+label%3Aclang%3Abounds-safety).
Patches for low-priority work that does not directly unblock upstreaming can
still slow upstreaming down, because each one takes time to review and to
reconcile with the downstream implementation.

The full `-fbounds-safety` model lives in the
[swiftlang/llvm-project](https://github.com/swiftlang/llvm-project) fork, which
is downstream of `llvm/llvm-project` and merges from it automatically. An
upstream change in this area can therefore cause conflicts downstream, so
upstream patches must also work with the full model downstream:

- Before merging a patch into `llvm/llvm-project`, cherry-pick it to the
  `next` branch of `swiftlang/llvm-project` and check that it works with the
  full `-fbounds-safety` model. Trivial updates may not need this.
- For more complex changes, the downstream may need to put the upstream
  behavior behind a flag, and then migrate to it gradually.

## Standardization

We are engaging with WG14, the C standard committee, with the long-term goal of
exploring bounds safety as an opt-in safety mode for C.

A first step is late parsing (retroactive scoping), which C currently lacks.
Bounds annotations often refer to program entities declared later, such as a
struct field or a later parameter. The following proposals introduce late
parsing for these cases:

- [N3796: Dependent attributes, v2](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3796.pdf):
  late parsing for attributes that describe a relationship with another
  program entity.
- [N3990: Forward Parameter References through Retroactive Scoping, v3](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3990.pdf):
  late parsing for array parameters, so that their sizes can be written in
  terms of other parameters of the same function.

## RFCs

- [RFC: Enforcing Bounds Safety in C (-fbounds-safety)](https://discourse.llvm.org/t/rfc-enforcing-bounds-safety-in-c-fbounds-safety/70854)
- [RFC: Bounds Safety in C Syntax Compatibility with GCC](https://discourse.llvm.org/t/rfc-bounds-safety-in-c-syntax-compatibility-with-gcc/85885)
- [RFC: Forward-referencing a struct member within bounds annotations](https://discourse.llvm.org/t/rfc-forward-referencing-a-struct-member-within-bounds-annotations/85510)

## Talks

- [2023 EuroLLVM - Keynote: "-fbounds-safety": Enforcing bounds safety for production C code](https://www.youtube.com/watch?v=RK9bfrsMdAM)
- [2025 EuroLLVM - Adopting -fbounds-safety in practice](https://www.youtube.com/watch?v=rlevLn831R4)
