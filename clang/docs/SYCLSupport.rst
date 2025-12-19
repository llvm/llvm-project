============
SYCL Support
============

.. contents::
   :local:

Introduction
============
The `SYCL 2020 specification <SYCL-2020_>`_ defines a single-source programming
model and C++ run-time library interface for writing portable programs that
support heterogeneous devices including GPUs, CPUs, and accelerators.
The specification is intended to allow for a wide range of implementation
possibilities, examples of which include:

- A SYCL run-time library written in standard C++ that executes kernels on a
  homogeneous set of host and device processors, each of which can execute
  common compiled code from shared memory.
- A SYCL run-time library that executes kernels on a heterogeneous set of
  device processors for which each kernel is pre-compiled for each supported
  device processor (Ahead-Of-Time (AOT) compilation) or for a family of device
  processors (Just-In-Time (JIT) compilation).

Since Clang is a conforming implementation of the C++ standard, no additional
features are required for support of the first implementation strategy.
This document details the core language features Clang provides for use by
SYCL run-time libraries that use the second implementation strategy.

.. _SYCL-2020:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html


Example Usage
=============
SYCL is designed as an extension of C++ rather than as a distinct programming
language.
SYCL support is enabled with the `-fsycl <opt-fsycl_>`_ option.

.. code-block:: sh

   clang++ -c -fsycl source-file.cpp

The choice of which target devices will be supported is made at compile time.
By default, SYCL source files will be compiled with support for a host target
dependent set of target devices.
For example, when compiling for a ``x86_64-unknown-linux-gnu`` host target,
target support will be enabled for ``spirv64-unknown-unknown`` devices.
The set of supported target devices can be specified via a comma separated list
of target triples with the `--offload-targets= <opt-offload-targets_>`_ option.
The following Clang invocation enables support for AMD, NVIDIA, and Intel GPU
targets.

.. code-block:: sh

   clang++ -c -fsycl \
     --offload-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda,spirv64-unknown-unknown \
     source-file.cpp

Object files built with the `-fsycl <opt-fsycl_>`_ option contain device
images that require additional processing at link time.
Programs linked with such object files must also be linked using the
``clang++`` driver and the `-fsycl <opt-fsycl_>`_ option.

.. code-block:: sh

   clang++ -fsycl example.o source-file.o -o example

.. _opt-fsycl:
   https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fsycl
.. _opt-offload-targets:
   https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-offload-targets


Compilation Model
=================
`SYCL 2020 section 5.1, "Offline compilation of SYCL source files" <SYCL-2020-5.1_>`_
acknowledges two compilation models.

- Single-source Multiple Compiler Pass (`SMCP`_) describes a compilation model
  in which source code is separately parsed and analyzed for the host target
  and each device target.

- Single-source Single Compiler Pass (`SSCP`_) describes a compilation model
  in which source code is parsed and analyzed once with code generation
  performed separately for the host target and each device target.

Clang only supports the `SMCP`_ compilation model currently, but the SYCL
language support features have been designed to allow for support of the
`SSCP`_ compilation model to be added in the future.

By default, SYCL source files are compiled for the host target and for each
device target.
In some cases, it is useful to restrict compilation to just the host target or
just the device targets; the `-fsycl-host-only <opt-fsycl-host-only_>`_ and
`-fsycl-device-only <opt-fsycl-device-only_>`_ options are available for these
purposes.

.. _SMCP:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:smcp
.. _SSCP:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:sscp
.. _SYCL-2020-5.1:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_offline_compilation_of_sycl_source_files
.. _opt-fsycl-host-only:
   https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-offload-host-only
.. _opt-fsycl-device-only:
   https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-offload-device-only


Supported Targets
=================
Support for SYCL is still in the implementation phase, but all targets
supported by the `--offload-targets= <opt-offload-targets_>`_ option
are intended to eventually be supported.


Predefined Macros
=================
`SYCL 2020 section 5.6, "Preprocessor directives and macros" <SYCL-2020-5.6_>`_
specifies macros that a SYCL implementation is required to provide.
Most such macros are defined by the SYCL run-time library and require inclusion
of the ``<sycl/sycl.hpp>`` header file.
The following macros are conditionally predefined by the compiler.

.. list-table::
   :header-rows: 1

   * - Macro
     - Description
   * - ``__SYCL_DEVICE_ONLY__``
     - Predefined by a `SMCP`_ implementation during device compilation (but not
       during host compilation).
   * - ``__SYCL_SINGLE_SOURCE__``
     - Predefined by a `SSCP`_ implementation during (host and device)
       compilation.

Since Clang only supports the `SMCP`_ compilation model currently, the
``__SYCL_SINGLE_SOURCE__`` macro is never predefined.

.. _SYCL-2020-5.6:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_preprocessor_directives_and_macros


Language Support
================
`SYCL 2020 section 3.12.3, "Library-only implementation" <SYCL-2020-3.12.13_>`_
notes the intent that the SYCL specification be implementable as a C++ library
with no requirements beyond a compiler that conforms to the C++17 standard.
The SYCL specification therefore does not specify extensions to the C++ core
language and a library-only implementation will work with Clang without any
core language extensions.
Clang provides the features described in this section to facilitate capabilities
that are not possible with a library-only SYCL implementation.

.. _SYCL-2020-3.12.13:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_library_only_implementation


.. _sect-sycl_kernel_entry_point:

The ``[[clang::sycl_kernel_entry_point]]`` Attribute
----------------------------------------------------
This attribute is intended for use in the implementation of SYCL run-time
libraries and should not be used directly by programmers.

The `sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_ attribute
coordinates interaction between Clang and the SYCL run-time library to
facilitate code generation and the execution of a SYCL kernel on a device
that potentially uses an instruction set architecture different from the host.
Consider the following call to the ``sycl::handler::single_task()`` SYCL
kernel invocation function.

.. code-block:: C++

   struct KN;
   void f(sycl::handler &h, sycl::stream &sout, int i) {
     h.single_task<KN>([=] {
       sout << "The value of i is " << i << "\n";
     });
   }

The SYCL kernel is defined by the lambda expression passed to the
``single_task()`` function and is identified by the ``KN`` type passed as the
first template argument.
See
`SYCL 2020 section 4.9.4.2, "SYCL functions for invoking kernels" <SYCL-2020-4.9.4.2_>`_
and
`SYCL 2020 section 5.2, "Naming of kernels" <SYCL-2020-5.2_>`_
for more details.

The `SMCP`_ and `SSCP`_ compilation models require that code generation be
performed for each SYCL kernel for each target device.
The `sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_ attribute
provides the means for the SYCL run-time library, which provides the
definition of all SYCL kernel invocation functions, to inform Clang of a SYCL
kernel invocation.
This is accomplished by including a call to a function declared with the
attribute in the definition of a SYCL kernel invocation function.
For example:

.. code-block:: C++

   namespace sycl {
     class handler {
       template <typename KernelName, typename KernelType>
       [[clang::sycl_kernel_entry_point(KernelName)]]
       void kernel_entry_point(KernelType kernelFunc) {
         kernelFunc();
       }
     public:
       template <typename KernelName, typename KernelType>
       void single_task(const KernelType &kernelFunc) {
         kernel_entry_point<KernelName>(kernelFunc);
       }
     };
   }

The arguments of the call to ``kernel_entry_point()`` constitute the parameters
of a SYCL kernel.
The body of the ``kernel_entry_point()`` function contains the statements
required to execute the SYCL kernel (e.g., an invocation of the call operator of
the SYCL kernel object as in this example).
The call to ``kernel_entry_point()`` in ``single_task()`` establishes a common
point of SYCL kernel invocation for Clang and the SYCL run-time library.
With that point established, the tasks required to actually execute a SYCL
kernel are delegated according to the following division of responsibilities.

Clang is responsible for:

- Validating that all kernel argument types (e.g., the deduced parameter types
  of the ``kernel_entry_point()`` function above) satisfy the requirements
  specified in
  `SYCL 2020 section 4.12.4, "Rules for parameter passing to kernels" <SYCL-2020-4.12.4_>`_.
- Informing the SYCL run-time library of the presence of subobjects of SYCL
  types that require special handling within kernel arguments.
- Generating an offload kernel entry point function for each SYCL kernel for
  each target device, generating a name for it derived from the SYCL kernel
  name, and informing the SYCL run-time library of the generated name.

The SYCL run-time library is responsible for:

- Selecting a device on which to execute the kernel.
- Copying the SYCL kernel object and any other kernel arguments to the device.
- Informing Clang of additional parameters required for the offload kernel
  entry point based on the presence of subobjects of SYCL types that require
  special handling within kernel arguments.
- Scheduling execution of the offload kernel entry point function on the
  selected device.

The SYCL run-time library tasks are expected to be performed in conjunction
with an offload backend such as liboffload, OpenCL, CUDA, HIP, or Level Zero;
their details are out of scope for this document.

The above division of responsibilities requires coordination.
The call to a function declared with the
`sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_ attribute causes
two primary side effects:

- The generation of an offload kernel entry point function.
- An implicit call to a SYCL run-time library provided template named
  ``sycl_kernel_launch`` (which may be a function template or a variable
  template of a type with a member call operator).

The offload kernel entry point function is generated with a target dependent
calling convention for each device target.
The function parameters and function body are initially copied from the function
declared with the `sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_
attribute, but may be augmented by information provided by the SYCL run-time
library as described below.
The function name is an implementation detail subject to change, but
incorporates the SYCL kernel name in order to ensure that a unique name is
deterministically generated for each SYCL kernel.

The call to the ``sycl_kernel_launch`` template effectively replaces the call
to the `sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_ attributed
function.
This implicit call serves several purposes:

- It informs the SYCL run-time library of the name of the offload kernel entry
  point function to be used to execute the kernel on the selected device.
- It informs the SYCL run-time library of the presence of subobjects of the
  kernel arguments that require special handling.

See the 
:ref:`sycl_special_kernel_parameter <sect-sycl_special_kernel_parameter>`
section regarding SYCL types that require special handling in kernel arguments.

The call to the ``sycl_kernel_launch`` template passes the name of the generated
offload kernel entry point function, the kernel arguments, and, for each
subobject of a kernel argument that requires special handling, a reference to
that subobject.
For reasons explained further below, the ``sycl_kernel_launch`` template needs
to know which of its arguments correspond to direct kernel arguments and which
correspond to references to special subobjects within the direct kernel
arguments.
Because there may be multiple kernel arguments with multiple subobjects that
require special handling, and because C++17 does not support function templates
with multiple function parameter packs, an idiom is used to pass the kernel
arguments and special subobjects separately.
This idiom is best explained by way of an example.

Consider the earlier example of a call to ``single_task()`` that passes a
lambda that captures variables of type ``int`` and ``std::stream``.
``std::stream`` is an example of a SYCL type that requires special handling in
kernel arguments.
The call to ``kernel_entry_point<KernelName>(kernelFunc)`` in the implementation
of ``single_task()`` results in an implicit call to ``sycl_kernel_launch`` that
looks similar to the following (the access to the captured copy of ``sout`` via
``kernelFunc.sout`` is not valid C++ syntax, but the compiler can generate such
accesses).

.. code-block:: C++

   sycl_kernel_launch<KernelName>("kernel-entry-point", kernelFunc)(kernelFunc.sout)

The SYCL kernel name type, ``KernelName``, is passed as an explicit template
type argument for convenient use by the SYCL run-time library if desired.
The first function argument is the name of the offload kernel entry point
function generated for the SYCL kernel denoted by ``KernelName``.
This argument names the function that the SYCL run-time library, in conjunction
with an offload backend, needs to resolve and execute on the selected device
in order to execute the SYCL kernel.
The remaining arguments are the kernel arguments for which there is just one in
this case; ``kernelFunc``.
Kernel arguments that contain a special subobject are passed as lvalues; those
that don't are passed as xvalues (and may therefore be moved from though such
types are unlikely to have optimized move operations since they are required to
be bit-copyable types).

The result of the ``sycl_kernel_launch`` call must be a function object (e.g.,
the result of a lambda expression, potentially one that captures references to
the kernel arguments).
The special subobjects are then passed as lvalues in an invocation of the
resulting function object.
This is the idiom mentioned earlier; this effectively allows for a call with
two function parameter packs.
A SYCL run-time library is expected to define the ``sycl_kernel_launch``
template in a form similar to this:

.. code-block:: C++

   namespace sycl {
     class handler {
       template <typename KernelName, typename... KernelArgs>
       auto sycl_kernel_launch(const char *entryPointName, KernelArgs &... args) {
         return [&] (auto &... subobjects) {
           // Process the kernel arguments and special subobjects, schedule
           // execution of 'entryPointName' on the selected device, and
           // return a type-list object specifying additional parameters to
           // add to the offload kernel entry point function (see below).
           return detail::type_list<...>{};
         };
       }
       ...
     };
  }

The above call to a ``sycl_kernel_launch`` template suffices for Clang to inform
the SYCL run-time library of the name of the generated offload kernel entry
point function to be executed, the kernel arguments to be passed to it, and the
subobjects of those kernel arguments that require special handling.
This is all the information the SYCL run-time library needs from Clang to
fulfill its requirements.

The special handling required for kernel arguments is an implementation detail
of the SYCL run-time library, but may require additional arguments to be passed
to the SYCL kernel.
This means that additional parameters may need to be added to the offload kernel
entry point function that Clang generates.
The SYCL run-time library informs Clang of the additional parameters by
returning a type-list object from the function object returned by the call to
``sycl_kernel_launch``.
A type-list object is a (possibly empty) object with a class template
specialization type; the template arguments for the specialization indicate
the additional parameters, if any, required for the kernel entry point.
The following class is suitable for type-list objects.

.. code-block:: C++

   namespace sycl {
     namespace detail {
       template <typename...>
       class type_list {};
     }
   }

Continuing with the earlier example, assume that the ``sycl::stream`` type holds
a reference to a buffer and that the SYCL run-time implementation uses an
offload backend that requires the buffer to be passed as individual kernel
arguments of type ``buffer_t*`` and ``int``.
The implicit call to ``sycl_kernel_launch`` informed the SYCL run-time library
of the captured ``sycl::stream`` variable, ``sout``, by passing a reference to
the capture to the function object returned by ``sycl_kernel_launch``.
The SYCL run-time library is now obligated to report the additional required
parameters by returning an appropriate type-list from that function object.
In general, computing the type-list requires metaprogramming to inspect all of
the special subobject types.
In this example, that metaprogramming would ultimately result in an object of
type ``detail::type_list<buffer_t*, int>`` being returned.
The types of the template arguments of the type of the returned object direct
Clang to add additional parameters of type ``buffer_t*`` and ``int`` to the
kernel entry point it generates.

With the signature of the entry point function now known, there is just one
remaining bit of information that Clang needs to be informed of; what to do
with the additional parameters in the body of the generated kernel entry point.
The required handling of these is delegated to a
``sycl_handle_special_kernel_parameters`` template that is implicitly called
in similar fashion to the ``sycl_kernel_launch`` template.
The direct arguments are the references to the special subobjects of the
kernel parameters and ``sycl_handle_special_kernel_parameters`` is required
to return a function object with a call operator that can receive the
additional parameters.
For example:

.. code-block:: C++

   namespace sycl {
     class handler {
       template <typename KernelName, typename... Subobjects>
       static auto sycl_handle_special_kernel_parameters(Subobjects &... sos) {
         return [&] (auto &... extraParams) -> void {
           // Process the special subobjects in order consuming elements of
           // 'extraParmas' as needed to update them.
         };
       }
       ...
     };
  }

With Clang now informed of the extra parameter handling necessitated by kernel
arguments with special subobjects, it has all the information needed to
generate the offload kernel entry point function.
The entry point function generated for the earlier example would look something
like the following (again, the use of ``kernelFunc.sout`` to access the captured
variable stored in ``kernelFunc`` is not valid C++ since captured variables
don't have names, but the intent should be clear; ``kernel-entry-point`` and
``lambda-from-f`` are exposition only names).

.. code-block:: C++

   void kernel-entry-point(lambda-from-f kernelFunc, buffer_t* X, int Y) {
     sycl_handle_special_kernel_parameters(kernelFunc.sout)(X, Y);
     kernelFunc();
   }


.. _attr-sycl_kernel_entry_point:
   https://clang.llvm.org/docs/AttributeReference.html#sycl-kernel-entry-point
.. _SYCL-2020-3.13.1:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable
.. _SYCL-2020-4.9.4.2:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:invokingkernels
.. _SYCL-2020-4.12.4:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:kernel.parameter.passing
.. _SYCL-2020-5.2:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:naming.kernels


.. _sect-sycl_special_kernel_parameter:

The ``[[clang::sycl_special_kernel_parameter]]`` Attribute
----------------------------------------------------------
This attribute is intended for use in the implementation of SYCL run-time
libraries and should not be used directly by programmers.

`SYCL 2020 section 4.12.4, "Rules for parameter passing to kernels" <SYCL-2020-4.12.4_>`_
specifies that objects of some SYCL types may be passed as kernel arguments
(including as data members or captures of SYCL kernel objects) even if their
class definitions do not satisfy the device copyable requirements specified in
`SYCL 2020 section 3.13.1, "Device copyable" <SYCL-2020-3.13.1_>`_.
The SYCL run-time library provides the definition of these types and is thus
responsible for managing the details of how they are transferred to a device
and how they are received as parameters of the offload kernel entry point
function.

Since C++17 lacks reflection features that would allow the SYCL run-time
library to identify use of these types for data members, captures, or base
classes of kernel argument types, the SYCL run-time library is dependent on
Clang to inform it of the presence of subobjects of these types within kernel
arguments.
The SYCL run-time library indicates which SYCL types require special handling
by declaring them with the
`sycl_special_kernel_parameter <attr-sycl_special_kernel_parameter_>`_
attribute.

When translating calls to functions declared with the
`sycl_kernel_entry_point <attr-sycl_kernel_entry_point_>`_ attribute, Clang
inspects each kernel argument type to look for data members, captures, and base
classes, that have a type declared with the
`sycl_special_kernel_parameter <attr-sycl_special_kernel_parameter_>`_
attribute.
For each such occurrence, Clang passes a reference to the associated subobject
when constructing the implicit call to the ``sycl_kernel_launch`` template.
See the 
:ref:`sycl_kernel_entry_point <sect-sycl_kernel_entry_point>` section for more
details.

For example, kernel arguments that contain a subobject of
``sycl::local_accessor`` type usually cannot be bit-copied to device memory for
use by a SYCL kernel; additional support from the offload backend is generally
required to manage their associated memory.
A SYCL run-time library implementation might therefore declare this type
similar to the following.

.. code-block:: C++

   namespace sycl {
     template <typename DataT, int Dimensions = 1>
     class [[clang::sycl_special_kernel_parameter]] local_accessor {
       ...
     };
   }

.. _attr-sycl_special_kernel_parameter:
   https://clang.llvm.org/docs/AttributeReference.html#sycl-special-kernel-parameter


The ``[[clang::sycl_external]]`` Attribute
------------------------------------------
This attribute is intended for use in the implementation of SYCL run-time
libraries and should not be used directly by programmers.

The `sycl_external <attr-sycl_external_>`_ attribute implements the semantics
required for the ``SYCL_EXTERNAL`` macro specified in
`SYCL 2020 section 5.10.1, "SYCL functions and member functions linkage" <SYCL-2020-5.10.1_>`_.
A function defined with this attribute is emitted for each device target
regardless of whether it is ODR-used.

The ``SYCL_EXTERNAL`` macro should be defined by the ``<sycl/sycl.hpp>`` header
file or one that it includes.

.. code-block:: C++

   #define SYCL_EXTERNAL [[clang::sycl_external]]

.. _attr-sycl_external:
   https://clang.llvm.org/docs/AttributeReference.html#sycl-external
.. _SYCL-2020-5.10.1:
   https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:syclexternal
