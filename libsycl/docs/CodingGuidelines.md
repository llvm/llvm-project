# Libsycl Coding Standards

:::{contents}
:local: true
:::

## Introduction

The `libsycl` project follows the
[LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) with
exceptions as described in this document.

### Naming

#### Names of Macros, Types, Functions, Variables, and Enumerators

Entities specified by the SYCL specification are named as required by the SYCL
specification. Names of all other entities follow the guidance in the LLVM
Coding Standards.

#### Names of Files and Directories

- **Directory Names** should be in snake case (e.g. `test_e2e`) except in
  cases where LLVM project wide conventions are used. For example, LIT tests
  often use an `Inputs` directory to hold files that are used by tests but
  that should be excluded from test discovery.
- **File Names in snake case** should be used for all C++ implementation files.
  For example files in directories `include`, `src`, `test`, `utils`,
  and `tools` should be named in snake case.
- **File Names in camel case** should be used for most other files. For example
  files in directories `cmake/modules` and `docs` should be named in camel
  case.

### Extension Naming Policy

The SYCL 2020 specification gives every vendor extension a vendor string. The
extension's APIs are declared in the `sycl::ext::<vendor>` namespace, its
feature-test macro is `SYCL_EXT_<VENDOR>_<NAME>`, and any enumerators or members
it adds to core SYCL classes (for example aspects) are prefixed with
`ext_<vendor>_`.

The vendor string tells users who defines the behavior of the feature. libsycl
picks it by that rule, not by who wrote the implementation.

| Origin of the extension                                       | Vendor string                                  |
| ------------------------------------------------------------- | ---------------------------------------------- |
| Not defined by any vendor: designed in the LLVM community     | `llvm`                                         |
| Defined by another vendor and implemented as specified        | The original one (e.g. `oneapi`, `intel`)      |
| Defined by another vendor, but libsycl intentionally deviates | `llvm`, only after the deviation is justified  |
| Ratified by Khronos                                           | `khr`                                          |

#### Community Extensions

An extension that the LLVM community designs itself uses the `llvm` vendor
string, for example `sycl::ext::llvm::<name>` with the feature-test macro
`SYCL_EXT_LLVM_<NAME>`. The community determines what such a feature is, so its
specification must be reviewed and committed together with the implementation.

#### Extensions Adopted From Another Vendor

When libsycl implements an extension defined by another vendor with the intent
to match it, the extension keeps its original name. For example, the
`sycl_ext_oneapi_<name>` extension from intel/llvm is implemented as
`sycl::ext::oneapi::<name>` with the feature-test macro
`SYCL_EXT_ONEAPI_<NAME>`. The original vendor's specification defines the
feature:

- Any behavioral difference from that specification is a libsycl bug.
- The implementation refers to the specification and to the revision of it that
  is implemented.
- The feature-test macro is defined, with the value of the implemented revision,
  only once that revision is fully implemented. Until then, the missing parts
  are listed in `libsycl/docs/index.md`.
- Differences that the specification does not make observable, such as
  implementation details, unspecified behavior, or diagnostic wording, are not
  deviations.
- If the specification appears to be wrong, the problem is raised with the
  vendor that owns it rather than fixed only in libsycl.
- New revisions of the specification are followed. An incompatible revision is
  implemented by updating to it and to its feature-test macro value.

Both `oneapi` and `intel` extensions are defined by Intel in intel/llvm and
follow this rule. `oneapi` extensions are designed to be device-agnostic, while
`intel` extensions expose features of Intel hardware or backends. The vendor
string of an `intel` extension stays `intel` even when libsycl implements it on
other devices, because it names the owner of the definition, not the hardware.

#### Deviating From Another Vendor's Extension

A deviation is any intentional change to the specified behavior: a different API
shape, different semantics, or a reduced or extended scope. Before deviating,
question why and whether it is a good idea. Two features with nearly the same
name but different behavior confuse users and force them to write more `#if`
blocks to use the feature portably. Proposing the change to the owning vendor is
preferred.

If the deviation is still justified, the extension is spelled with the `llvm`
vendor string, e.g. `sycl::ext::llvm::<name>`, to signal that it is not
`sycl::ext::oneapi::<name>`. It is then a community extension with its own
specification. The original vendor's spelling must not be used for the deviating
extension, including as an alias.

#### Khronos Extensions

Extensions ratified by Khronos use the `khr` vendor string and are implemented
as specified, following the same rules as extensions adopted from another
vendor. When a vendor extension that libsycl implements is promoted to a `khr`
extension, libsycl implements the `khr` extension, and the vendor version is
deprecated.

#### Rationale

C++ attributes follow the same model. When GCC defines an attribute and Clang
implements it with the same behavior, Clang accepts the `gnu` spelling, as in
`struct [[gnu::packed]] S`, because GCC defines what the attribute means.
Adding it only as `[[clang::packed]]` would leave users unsure how
`gnu::packed` and `clang::packed` differ and would make portable code need
more `#if` blocks. Clang uses its own `clang` namespace for attributes it
defines itself.

