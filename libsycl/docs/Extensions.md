# SYCL extensions

:::{contents}
:local: true
:::

This page lists the SYCL extensions implemented by libsycl and links to the specification that defines each of them.
As described in the [Extension Naming Policy](CodingGuidelines.md#extension-naming-policy), an extension adopted from another vendor is implemented as that vendor's specification defines it.
An extension may be listed here before it is fully implemented, but the feature-test macro won't be defined until it is fully implemented.

## Vendor extensions

|Extension|Implemented revision|Feature-test macro|
|---|---|---|
|[`sycl_ext_oneapi_backend_level_zero`]|(not fully implemented)|`SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO`, not defined|

[`sycl_ext_oneapi_backend_level_zero`]: https://github.com/intel/llvm/blob/b75e13a4e84774b47334b267747ca6d46e487bd7/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md
