.. _index:

==============================
"libsycl" SYCL runtime library
==============================

Overview
========

libsycl is an implementation of the SYCL runtime library as defined by the
`SYCL 2020 specification <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_.

libsycl runtime library and headers require C++17 support or higher.

Current Status
==============

The implementation is in the very early stages of upstreaming. The first milestone is to get
support for a simple SYCL application with device code using Unified Shared Memory:

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
   
     for (int i = 0; i < dataSize; ++i)
       if (dataPtr[i] != i) return 1;
   
     free(dataPtr, q);
   
     return 0;
   }

This requires at least partial support of the following functionality on the libsycl side:
  * sycl::platform class
  * sycl::device class
  * sycl::context class
  * sycl::queue class
  * sycl::handler class
  * sycl::id and sycl::range classes
  * Unified shared memory allocation/deallocation
  * Program manager, an internal component for retrieving and using device images from the fat binary
  