# SYCL extensions

:::{contents}
:local: true
:::

This page lists the SYCL extensions implemented by libsycl and links to the
specification that defines each of them. As described in the
[Extension Naming Policy](CodingGuidelines.md#extension-naming-policy), an
extension adopted from another vendor is implemented as that vendor's
specification defines it, and its feature-test macro is defined only once the
listed revision is fully implemented.

## Vendor extensions

|Extension|Specification|Implemented revision|Feature-test macro|
|---|---|---|---|
|`sycl_ext_oneapi_backend_level_zero`|[intel/llvm](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md)| - |`SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO`, not defined|
