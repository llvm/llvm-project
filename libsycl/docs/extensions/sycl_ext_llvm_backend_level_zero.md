# Level Zero backend specification

:::{contents}
:local: true
:::

## 1. Introduction

This document is the SYCL backend specification associated with the
`sycl::backend::ext_llvm_level_zero` enumerator, as required by the SYCL 2020
specification section 4.1 "Backends". The backend is built on top of the Level
Zero API and is reached through the Level Zero plugin of the offload library.
The currently supported targets are Intel GPUs. For the Level Zero API see
<https://oneapi-src.github.io/level-zero-spec/level-zero/latest/index.html>.

The backend follows the naming scheme for extensions from the SYCL 2020
specification section 6.3.7 "Adding a backend", with `llvm` being the
`<vendorstring>` of this implementation. An implementation that makes this
backend available predefines the `SYCL_EXT_LLVM_BACKEND_LEVEL_ZERO` macro to
one of the values below. The macro doubles as a feature test macro, so an
application can test its value to determine which parts of this specification
are available.

|Value|Description|
|---|:---|
|1|Initial version. Backend selection and queries only, no interoperability.

## 2. Prerequisites

The Level Zero loader and drivers need to be installed on the system for the
offload runtime to recognize and enable the Level Zero backend.

The offload library libsycl is built on top of must be built with its Level Zero
plugin enabled, which is what the build steps in [index](../index.md) configure
through `-DLIBOMPTARGET_PLUGINS_TO_BUILD=level_zero`.

## 3. User-visible Level Zero backend selection (and the default backend)

The Level Zero backend is added to the `sycl::backend` enumeration:

```c++
namespace sycl {
enum class backend : /* unspecified */ {
  // ...
  ext_llvm_level_zero,
  // ...
};
} // namespace sycl
```

libsycl does not provide a way for the user to request a specific backend. Every
platform reported by the offload library, except for the host one, is exposed to
the application, and the backend serving a platform is chosen by the
implementation: it is `backend::ext_llvm_level_zero` for the GPU devices
supported by the installed Level Zero runtime. When several devices are equally
suitable for a device selector, the ones served by the Level Zero backend are
preferred.

The serving backend of a `platform`, `device`, `context`, `queue` or `event`
can be queried with the `get_backend()` member function of the corresponding
class.

Any use of the `backend` enumeration is non-generic SYCL, so, as described in
the SYCL 2020 specification section 4.2 "Generic vs non-generic SYCL", such use
must be guarded with the macro associated with the backend:

```c++
#ifdef SYCL_EXT_LLVM_BACKEND_LEVEL_ZERO
  if (Queue.get_backend() == sycl::backend::ext_llvm_level_zero) {
    // Level Zero specific code path.
  }
#endif
```
