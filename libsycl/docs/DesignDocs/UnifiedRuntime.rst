=====================
Unified Runtime
=====================

.. contents::
   :local:

.. _unified runtime:

The Unified Runtime (UR) project serves as an interface layer between the SYCL
runtime and the device-specific runtime layers which control execution on
devices. SYCL RT utilizes its C API, loader library, and the adapter libraries
that implement the API for various backends.

The SYCL runtime accesses the UR API via the Adapter object. Each Adapter
object owns a ``ur_adapter_handle_t``, which represents a UR backend (e.g. OpenCL,
Level Zero, etc).

For detailed information about the UR project including
the API specification see the `Unified Runtime Documentation
<https://oneapi-src.github.io/unified-runtime/core/INTRO.html>`__. You
can find the Unified Runtime repo `here
<https://github.com/oneapi-src/unified-runtime>`__.
