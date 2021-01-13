.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

.. contents::
   :local:

==================
OpenCL Support
==================

Clang fully supports all OpenCL C versions from 1.1 to 2.0.

Please refer to `Bugzilla
<https://bugs.llvm.org/buglist.cgi?component=OpenCL&list_id=172679&product=clang&resolution=--->`__
for the most up to date bug reports.


C++ for OpenCL Implementation Status
====================================

Bugzilla bugs for this functionality are typically prefixed
with '[C++]'.

Differences to OpenCL C
-----------------------

TODO!

Missing features or with limited support
----------------------------------------

- Use of ObjC blocks is disabled.

- Global destructor invocation is not generated correctly.

- Initialization of objects in `__constant` address spaces is not guaranteed to work.

- `addrspace_cast` operator is not supported.

Experimental features
=====================

Clang provides the following new WIP features for the developers to experiment
and provide early feedback or contribute with further improvements.
Feel free to contact us on `cfe-dev
<https://lists.llvm.org/mailman/listinfo/cfe-dev>`_ or via `Bugzilla
<https://bugs.llvm.org/>`__.

C++ libraries for OpenCL
------------------------

There is ongoing work to support C++ standard libraries from `LLVM's libcxx
<https://libcxx.llvm.org/>`_ in OpenCL kernel code using C++ for OpenCL mode.

It is currently possible to include `type_traits` from C++17 in the kernel
sources when the following clang extensions are enabled
``__cl_clang_function_pointers`` and ``__cl_clang_variadic_functions``,
see :doc:`LanguageExtensions` for more details. The use of non-conformant
features enabled by the extensions does not expose non-conformant behavior
beyond the compilation i.e. does not get generated in IR or binary.
The extension only appear in metaprogramming
mechanism to identify or verify the properties of types. This allows to provide
the full C++ functionality without a loss of portability. To avoid unsafe use
of the extensions it is recommended that the extensions are disabled directly
after the header include.

**Example of Use**:

The example of kernel code with `type_traits` is illustrated here.

.. code-block:: c++

  #pragma OPENCL EXTENSION __cl_clang_function_pointers : enable
  #pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable
  #include <type_traits>
  #pragma OPENCL EXTENSION __cl_clang_function_pointers : disable
  #pragma OPENCL EXTENSION __cl_clang_variadic_functions : disable

  using sint_type = std::make_signed<unsigned int>::type;

  __kernel void foo() {
    static_assert(!std::is_same<sint_type, unsigned int>::value);
  }

The possible clang invocation to compile the example is as follows:

   .. code-block:: console

     $ clang -cl-std=clc++  -I<path to libcxx checkout or installation>/include test.cl

Note that `type_traits` is a header only library and therefore no extra
linking step against the standard libraries is required.
