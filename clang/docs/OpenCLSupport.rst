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

Clang has complete support of OpenCL C versions from 1.0 to 2.0.

Clang also supports :ref:`the C++ for OpenCL kernel language <cxx_for_opencl_impl>`.

There is an ongoing work to support :ref:`OpenCL 3.0 <opencl_300>`.

There are also other :ref:`new and experimental features <opencl_experimenal>` available.

For general issues and bugs with OpenCL in clang refer to `Bugzilla
<https://bugs.llvm.org/buglist.cgi?component=OpenCL&list_id=172679&product=clang&resolution=--->`__.

.. _cxx_for_opencl_impl:

C++ for OpenCL Implementation Status
====================================

Clang implements language version 1.0 published in `the official
release of C++ for OpenCL Documentation
<https://github.com/KhronosGroup/OpenCL-Docs/releases/tag/cxxforopencl-v1.0-r1>`_.

Limited support of experimental C++ libraries is described in the :ref:`experimental features <opencl_experimenal>`.

Bugzilla bugs for this functionality are typically prefixed
with '[C++4OpenCL]' - click `here
<https://bugs.llvm.org/buglist.cgi?component=OpenCL&list_id=204139&product=clang&query_format=advanced&resolution=---&sh    ort_desc=%5BC%2B%2B4OpenCL%5D&short_desc_type=allwordssubstr>`_
to view the full bug list.


Missing features or with limited support
----------------------------------------

- Use of ObjC blocks is disabled and therefore the ``enqueue_kernel`` builtin
  function is not supported currently. It is expected that if support for this
  feature is added in the future, it will utilize C++ lambdas instead of ObjC
  blocks.

- IR generation for global destructors is incomplete (See:
  `PR48047 <https://llvm.org/PR48047>`_).

- There is no distinct file extension for sources that are to be compiled
  in C++ for OpenCL mode (See: `PR48097 <https://llvm.org/PR48097>`_)

.. _opencl_300:

OpenCL 3.0 Implementation Status
================================

The following table provides an overview of features in OpenCL C 3.0 and their
implementation status. 

+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Category                     | Feature                                                      | Status               | Reviews                                                                   |
+==============================+==============================================================+======================+===========================================================================+
| Command line interface       | New value for ``-cl-std`` flag                               | :good:`done`         | https://reviews.llvm.org/D88300                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Predefined macros            | New version macro                                            | :good:`done`         | https://reviews.llvm.org/D88300                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Predefined macros            | Feature macros                                               | :part:`worked on`    | https://reviews.llvm.org/D89869                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Generic address space                                        | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Builtin function overloads with generic address space        | :part:`worked on`    | https://reviews.llvm.org/D92004                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Program scope variables in global memory                     | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | 3D image writes including builtin functions                  | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | read_write images including builtin functions                | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | C11 atomics memory scopes, ordering and builtin function     | :part:`worked on`    | https://reviews.llvm.org/D92004 (functions only)                          |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Device-side kernel enqueue including builtin functions       | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Pipes including builtin functions                            | :part:`worked on`    | https://reviews.llvm.org/D92004 (functions only)                          |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| Feature optionality          | Work group collective functions                              | :part:`worked on`    | https://reviews.llvm.org/D92004                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| New functionality            | RGBA vector components                                       | :none:`unclaimed`    |                                                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| New functionality            | Subgroup functions                                           | :part:`worked on`    | https://reviews.llvm.org/D92004                                           |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+
| New functionality            | Atomic mem scopes: subgroup, all devices including functions | :part:`worked on`    | https://reviews.llvm.org/D92004 (functions only)                          |
+------------------------------+--------------------------------------------------------------+----------------------+---------------------------------------------------------------------------+

.. _opencl_experimenal:

Experimental features
=====================

Clang provides the following new WIP features for the developers to experiment
and provide early feedback or contribute with further improvements.
Feel free to contact us on `cfe-dev
<https://lists.llvm.org/mailman/listinfo/cfe-dev>`_ or via `Bugzilla
<https://bugs.llvm.org/>`__.

Fast builtin function declarations
----------------------------------

In addition to regular header includes with builtin types and functions using
``-finclude-default-header`` explained in :doc:`UsersManual`, clang
supports a fast mechanism to declare builtin functions with
``-fdeclare-opencl-builtins``. This does not declare the builtin types and
therefore it has to be used in combination with ``-finclude-default-header``
if full functionality is required.

**Example of Use**:

    .. code-block:: console
 
      $ clang -Xclang -finclude-default-header test.cl

Note that this is a frontend-only flag and therefore it requires the use of
flags that forward options to the frontend, e.g. ``-cc1`` or ``-Xclang``.

As this feature is still in experimental phase some changes might still occur
on the command line interface side.

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
