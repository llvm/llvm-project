=====================
Clang Offload Bundler
=====================

.. contents::
   :local:

.. _clang-offload-bundler:

Introduction
============

For heterogeneous single source programming languages, use one or more
``--offload-arch=<target-id>`` Clang options to specify the target IDs of the
code to generate for the offload code regions.

The tool chain may perform multiple compilations of a translation unit to
produce separate code objects for the host and potentially multiple offloaded
devices. The ``clang-offload-bundler`` tool may be used as part of the tool
chain to combine these multiple code objects into a single bundled code object.

The tool chain may use a bundled code object as an intermediate step so that
each tool chain step consumes and produces a single file as in traditional
non-heterogeneous tool chains. The bundled code object contains the code objects
for the host and all the offload devices.

A bundled code object may also be used to bundle just the offloaded code
objects, and embedded as data into the host code object. The host compilation
includes an ``init`` function that will use the runtime corresponding to the
offload kind (see :ref:`clang-offload-kind-table`) to load the offload code
objects appropriate to the devices present when the host program is executed.

.. _clang-bundled-code-object-layout:

Bundled Code Object Layout
==========================

The layout of a bundled code object is defined by the following table:

  .. table:: Bundled Code Object Layout
    :name: bundled-code-object-layout-table

    =================================== ======= ================ ===============================
    Field                               Type    Size in Bytes    Description
    =================================== ======= ================ ===============================
    Magic String                        string  24               ``__CLANG_OFFLOAD_BUNDLE__``
    Number Of Bundle Entries            integer 8                Number of bundle entries.
    1st Bundle Entry Code Object Offset integer 8                Byte offset from beginning of
                                                                 bundled code object to 1st code
                                                                 object.
    1st Bundle Entry Code Object Size   integer 8                Byte size of 1st code object.
    1st Bundle Entry ID Length          integer 8                Character length of bundle
                                                                 entry ID of 1st code object.
    1st Bundle Entry ID                 string  1st Bundle Entry Bundle entry ID of 1st code
                                                ID Length        object. This is not NUL
                                                                 terminated. See
                                                                 :ref:`clang-bundle-entry-id`.
    \...
    Nth Bundle Entry Code Object Offset integer 8
    Nth Bundle Entry Code Object Size   integer 8
    Nth Bundle Entry ID Length          integer 8
    Nth Bundle Entry ID                 string  1st Bundle Entry
                                                ID Length
    1st Bundle Entry Code Object        bytes   1st Bundle Entry
                                                Code Object Size
    \...
    Nth Bundle Entry Code Object        bytes   Nth Bundle Entry
                                                Code Object Size
    =================================== ======= ================ ===============================

.. _clang-bundle-entry-id:

Bundle Entry ID
===============

Each entry in a bundled code object (see
:ref:`clang-bundled-code-object-layout`) has a bundle entry ID that indicates
the kind of the entry's code object and the runtime that manages it.

Bundle entry ID syntax is defined by the following BNF syntax:

.. code::

  <bundle-entry-id> ::== <offload-kind> "-" <target-triple> [ "-" <target-id> ]

Where:

**offload-kind**
  The runtime responsible for managing the bundled entry code object. See
  :ref:`clang-offload-kind-table`.

  .. table:: Bundled Code Object Offload Kind
      :name: clang-offload-kind-table

      ============= ==============================================================
      Offload Kind  Description
      ============= ==============================================================
      host          Host code object. ``clang-offload-bundler`` always includes
                    this entry as the first bundled code object entry. For an
                    embedded bundled code object this entry is not used by the
                    runtime and so is generally an empty code object.

      hip           Offload code object for the HIP language. Used for all
                    HIP language offload code objects when the
                    ``clang-offload-bundler`` is used to bundle code objects as
                    intermediate steps of the tool chain. Also used for AMD GPU
                    code objects before ABI version V4 when the
                    ``clang-offload-bundler`` is used to create a *fat binary*
                    to be loaded by the HIP runtime. The fat binary can be
                    loaded directly from a file, or be embedded in the host code
                    object as a data section with the name ``.hip_fatbin``.

      hipv4         Offload code object for the HIP language. Used for AMD GPU
                    code objects with at least ABI version V4 when the
                    ``clang-offload-bundler`` is used to create a *fat binary*
                    to be loaded by the HIP runtime. The fat binary can be
                    loaded directly from a file, or be embedded in the host code
                    object as a data section with the name ``.hip_fatbin``.

      openmp        Offload code object for the OpenMP language extension.
      ============= ==============================================================

**target-triple**
  The target triple of the code object.

**target-id**
  The canonical target ID of the code object. Present only if the target
  supports a target ID. See :ref:`clang-target-id`.

Each entry of a bundled code object must have a different bundle entry ID. There
can be multiple entries for the same processor provided they differ in target
feature settings. If there is an entry with a target feature specified as *Any*,
then all entries must specify that target feature as *Any* for the same
processor. There may be additional target specific restrictions.

.. _clang-target-id:

Target ID
=========

A target ID is used to indicate the processor and optionally its configuration,
expressed by a set of target features, that affect ISA generation. It is target
specific if a target ID is supported, or if the target triple alone is
sufficient to specify the ISA generation.

It is used with the ``-mcpu=<target-id>`` and ``--offload-arch=<target-id>``
Clang compilation options to specify the kind of code to generate.

It is also used as part of the bundle entry ID to identify the code object. See
:ref:`clang-bundle-entry-id`.

Target ID syntax is defined by the following BNF syntax:

.. code::

  <target-id> ::== <processor> ( ":" <target-feature> ( "+" | "-" ) )*

Where:

**processor**
  Is a the target specific processor or any alternative processor name.

**target-feature**
  Is a target feature name that is supported by the processor. Each target
  feature must appear at most once in a target ID and can have one of three
  values:

  *Any*
    Specified by omitting the target feature from the target ID.
    A code object compiled with a target ID specifying the default
    value of a target feature can be loaded and executed on a processor
    configured with the target feature on or off.

  *On*
    Specified by ``+``, indicating the target feature is enabled. A code
    object compiled with a target ID specifying a target feature on
    can only be loaded on a processor configured with the target feature on.

  *Off*
    specified by ``-``, indicating the target feature is disabled. A code
    object compiled with a target ID specifying a target feature off
    can only be loaded on a processor configured with the target feature off.

There are two forms of target ID:

*Non-Canonical Form*
  The non-canonical form is used as the input to user commands to allow the user
  greater convenience. It allows both the primary and alternative processor name
  to be used and the target features may be specified in any order.

*Canonical Form*
  The canonical form is used for all generated output to allow greater
  convenience for tools that consume the information. It is also used for
  internal passing of information between tools. Only the primary and not
  alternative processor name is used and the target features are specified in
  alphabetic order. Command line tools convert non-canonical form to canonical
  form.

Target Specific information
===========================

Target specific information is available for the following:

*AMD GPU*
  AMD GPU supports target ID and target features. See `User Guide for AMDGPU Backend
  <https://llvm.org/docs/AMDGPUUsage.html>`_ which defines the `processors
  <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processors>`_ and `target
  features <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-target-features>`_
  supported.

Most other targets do not support target IDs.
