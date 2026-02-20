===========================
SYCL runtime implementation
===========================

.. contents::
   :local:

.. _index:

Current Status
==============

The implementation is in the very early stages of upstreaming. The first
milestone is to get
support for a simple SYCL application with device code using Unified Shared
Memory:

.. code-block:: c++

   #include <sycl/sycl.hpp>
   
   class TestKernel;
   
   int main() {
     sycl::queue q;
   
     const size_t dataSize = 32;
     int *dataPtr = sycl::malloc_shared<int>(32, q);
     for (int i = 0; i < dataSize; ++i)
       dataPtr[i] = 0;
   
     q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<TestKernel>(
           sycl::range<1>(dataSize),
           [=](sycl::id<1> idx) { dataPtr[idx] = idx[0]; });
     });
     q.wait();
   
     bool error = false;
     for (int i = 0; i < dataSize; ++i)
       if (dataPtr[i] != i) error = true;
   
     free(dataPtr, q);
   
     return error;
   }

This requires at least partial support of the following functionality on the
libsycl side:

* ``sycl::platform`` class
* ``sycl::device`` class
* ``sycl::context`` class
* ``sycl::queue`` class
* ``sycl::handler`` class
* ``sycl::id`` and ``sycl::range`` classes
* Unified shared memory allocation/deallocation
* Program manager, an internal component for retrieving and using device images
  from the multi-architectural binaries

Build steps
===========

To build LLVM with libsycl runtime enabled the following script can be used.

.. code-block:: console

  #!/bin/sh

  build_llvm=`pwd`/build-llvm
  installprefix=`pwd`/install
  llvm=`pwd`
  mkdir -p $build_llvm
  mkdir -p $installprefix

  cmake -G Ninja -S $llvm/llvm -B $build_llvm \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_INSTALL_UTILS=ON \
        -DCMAKE_INSTALL_PREFIX=$installprefix \
        -DLLVM_ENABLE_RUNTIMES="offload;openmp;libsycl" \
        -DCMAKE_BUILD_TYPE=Release \
        # must be default and configured in liboffload,
        # requires level zero, see offload/cmake/Modules/LibomptargetGetDependencies.cmake
        -DLIBOMPTARGET_PLUGINS_TO_BUILD=level_zero

  ninja -C $build_llvm install


Limitations
===========

Libsycl is not currently supported on Windows because it depends on liboffload
which doesn't currently support Windows.

TODO for added SYCL classes
===========================

* ``exception``: methods with context are not implemented, to add once context is ready
* ``platform``: deprecated info descriptor is not implemented (info::platform::extensions), to implement on RT level with ``device::get_info<info::device::aspects>()``
* ``device``:

  * ``get_info``: to find an efficient way to map descriptors to liboffload types, add other descriptors, add cache of info data
  * ``has(aspect)``: same as get_info
  * ``create_sub_devices``: partitioning is not supported by liboffload now, blocked
  * ``has_extension``: deprecated API, to implement on RT level with ``device::has``

* device selection: to add compatibility with old SYCL 1.2.1 device selectors, still part of SYCL 2020 specification

