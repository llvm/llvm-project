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

`libsycl` implements two kinds of SYCL extensions:

- **Khronos extensions** (KHR extensions) are ratified by the Khronos Group
  and published alongside the SYCL specification.
- **Vendor extensions** are defined by a single vendor, which may be the LLVM
  community itself.

The two kinds follow different naming rules, described below.

#### Khronos Extensions

KHR extensions are defined only by the Khronos Group. Their APIs are declared
in the `sycl::khr` namespace and their feature-test macros are named
`SYCL_KHR_<NAME>`.

If `libsycl` adopts a KHR extension, it is implemented according to the KHR
extension specification, following the same rules as extensions adopted from
another vendor (see below).

#### Vendor Extensions

`libsycl` follows the "Guidelines for portable extensions" described in
[Chapter 6](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#chapter.extensions)
of the SYCL 2020 specification. These guidelines require each vendor extension
to be named with a vendor string. For example, the extension's APIs are
declared in the `sycl::ext::<vendor>` namespace, its feature-test macro is
`SYCL_EXT_<VENDOR>_<NAME>`, and any enumerators or members it adds to core SYCL
classes (for example aspects) are prefixed with `ext_<vendor>_`.

The vendor string tells users who defines the behavior of the feature.

| Origin of the extension                                | Vendor string                                  |
| ------------------------------------------------------ | ---------------------------------------------- |
| Defined by the LLVM community                          | `llvm`                                         |
| Defined by another vendor and implemented as specified | The original vendor's (e.g. `oneapi`, `intel`) |

##### Community Extensions

An extension that the LLVM community designs itself uses the `llvm` vendor
string, for example `sycl::ext::llvm::<name>` with the feature-test macro
`SYCL_EXT_LLVM_<NAME>`. The community determines what such a feature is, so its
specification must be reviewed and committed before or together with the
implementation.

##### Extensions Adopted From Another Vendor

When `libsycl` implements an extension defined by another vendor with the intent
to match it, the extension keeps its original name. For example, the
`sycl_ext_oneapi_<name>` extension from the Intel DPC++ compiler is implemented
as `sycl::ext::oneapi::<name>` with the feature-test macro
`SYCL_EXT_ONEAPI_<NAME>`. The original vendor's specification defines the
feature:

- Any behavioral difference from that specification is a `libsycl` bug.
- The implementation refers to the specification and to the revision of it that
  is implemented.
- The feature-test macro is defined, with the value of the implemented revision,
  only once that revision is fully implemented.
- Differences that the specification does not make observable, such as
  implementation details, unspecified behavior, or diagnostic wording, are not
  deviations.
- If the specification appears to be wrong, the problem is raised with the
  vendor that owns it rather than fixed only in `libsycl`.
- When the vendor publishes a new revision of the specification, `libsycl` is
  updated to conform to it.

If the community does not agree with some part of a vendor's extension
specification, it works with the vendor to come to an agreement. If an
agreement cannot be reached, the community is free to draft a new extension
specification with different behavior. However, this extension uses the `llvm`
vendor string. `libsycl` never intentionally implements an extension from
another vendor in a way that deviates from that vendor's specification.

#### Rationale

C++ attributes follow the same model. When GCC defines an attribute and Clang
implements it with the same behavior, Clang accepts the `gnu` spelling, as in
`struct [[gnu::packed]] S`, because GCC defines what the attribute means.
Adding it only as `[[clang::packed]]` would leave users unsure how
`gnu::packed` and `clang::packed` differ and would make portable code need
more `#if` blocks. Clang uses its own `clang` namespace for attributes it
defines itself.

