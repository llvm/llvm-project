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

Internals Manual
================

This section acts as internal documentation for OpenCL features design
as well as some important implementation aspects. It is primarily targeted
at the advanced users and the toolchain developers integrating frontend
functionality as a component.

OpenCL Metadata
---------------

Clang uses metadata to provide additional OpenCL semantics in IR needed for
backends and OpenCL runtime.

Each kernel will have function metadata attached to it, specifying the arguments.
Kernel argument metadata is used to provide source level information for querying
at runtime, for example using the `clGetKernelArgInfo 
<https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf#167>`_
call.

Note that ``-cl-kernel-arg-info`` enables more information about the original
kernel code to be added e.g. kernel parameter names will appear in the OpenCL
metadata along with other information. 

The IDs used to encode the OpenCL's logical address spaces in the argument info
metadata follows the SPIR address space mapping as defined in the SPIR
specification `section 2.2
<https://www.khronos.org/registry/spir/specs/spir_spec-2.0.pdf#18>`_

OpenCL Specific Options
-----------------------

In addition to the options described in :doc:`UsersManual` there are the
following options specific to the OpenCL frontend.

.. _opencl_cl_ext:

.. option:: -cl-ext

Disables support of OpenCL extensions. All OpenCL targets provide a list
of extensions that they support. Clang allows to amend this using the ``-cl-ext``
flag with a comma-separated list of extensions prefixed with ``'+'`` or ``'-'``.
The syntax: ``-cl-ext=<(['-'|'+']<extension>[,])+>``,  where extensions
can be either one of `the OpenCL published extensions
<https://www.khronos.org/registry/OpenCL>`_
or any vendor extension. Alternatively, ``'all'`` can be used to enable
or disable all known extensions.

Note that this is a frontend-only flag and therefore it requires the use of
flags that forward options to the frontend e.g. ``-cc1`` or ``-Xclang``.

Example disabling double support for the 64-bit SPIR target:

   .. code-block:: console

     $ clang -cc1 -triple spir64-unknown-unknown -cl-ext=-cl_khr_fp64 test.cl

Enabling all extensions except double support in R600 AMD GPU can be done using:

   .. code-block:: console

     $ clang -cc1 -triple r600-unknown-unknown -cl-ext=-all,+cl_khr_fp16 test.cl

.. _opencl_fake_address_space_map:

.. option:: -ffake-address-space-map

Overrides the target address space map with a fake map.
This allows adding explicit address space IDs to the bitcode for non-segmented
memory architectures that do not have separate IDs for each of the OpenCL
logical address spaces by default. Passing ``-ffake-address-space-map`` will
add/override address spaces of the target compiled for with the following values:
``1-global``, ``2-constant``, ``3-local``, ``4-generic``. The private address
space is represented by the absence of an address space attribute in the IR (see
also :ref:`the section on the address space attribute <opencl_addrsp>`).

   .. code-block:: console

     $ clang -cc1 -ffake-address-space-map test.cl

Note that this is a frontend-only flag and therefore it requires the use of
flags that forward options to the frontend e.g. ``-cc1`` or ``-Xclang``.

OpenCL builtins
---------------

There are some standard OpenCL functions that are implemented as Clang builtins:

- All pipe functions from `section 6.13.16.2/6.13.16.3
  <https://www.khronos.org/registry/cl/specs/opencl-2.0-openclc.pdf#160>`_ of
  the OpenCL v2.0 kernel language specification. `

- Address space qualifier conversion functions ``to_global``/``to_local``/``to_private``
  from `section 6.13.9
  <https://www.khronos.org/registry/cl/specs/opencl-2.0-openclc.pdf#101>`_.

- All the ``enqueue_kernel`` functions from `section 6.13.17.1
  <https://www.khronos.org/registry/cl/specs/opencl-2.0-openclc.pdf#164>`_ and
  enqueue query functions from `section 6.13.17.5
  <https://www.khronos.org/registry/cl/specs/opencl-2.0-openclc.pdf#171>`_.

.. _opencl_addrsp:

Address spaces attribute
------------------------

Clang has arbitrary address space support using the ``address_space(N)``
attribute, where ``N`` is an integer number in the range specified in the
Clang source code. This addresses spaces can be used along with the OpenCL
address spaces however when such addresses spaces converted to/from OpenCL
address spaces the behavior is not governed by OpenCL specification.

An OpenCL implementation provides a list of standard address spaces using
keywords: ``private``, ``local``, ``global``, and ``generic``. In the AST and
in the IR each of the address spaces will be represented by unique number
provided in the Clang source code. The specific IDs for an address space do not
have to match between the AST and the IR. Typically in the AST address space
numbers represent logical segments while in the IR they represent physical
segments.
Therefore, machines with flat memory segments can map all AST address space
numbers to the same physical segment ID or skip address space attribute
completely while generating the IR. However, if the address space information
is needed by the IR passes e.g. to improve alias analysis, it is recommended
to keep it and only lower to reflect physical memory segments in the late
machine passes. The mapping between logical and target address spaces is
specified in the Clang's source code.

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
 
      $ clang -Xclang -fdeclare-opencl-builtins test.cl

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
