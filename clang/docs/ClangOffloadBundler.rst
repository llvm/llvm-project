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

Supported File Formats
======================
Several text and binary file formats are supported for bundling/unbundling. See
:ref:`supported-file-formats-table` for a list of currently supported formats.

  .. table:: Supported File Formats
     :name: supported-file-formats-table

     +--------------------+----------------+-------------+
     | File Format        | File Extension | Text/Binary |
     +====================+================+=============+
     | CPP output         |        i       |     Text    |
     +--------------------+----------------+-------------+
     | C++ CPP output     |       ii       |     Text    |
     +--------------------+----------------+-------------+
     | CUDA/HIP output    |       cui      |     Text    |
     +--------------------+----------------+-------------+
     | Dependency         |        d       |     Text    |
     +--------------------+----------------+-------------+
     | LLVM               |       ll       |     Text    |
     +--------------------+----------------+-------------+
     | LLVM Bitcode       |       bc       |    Binary   |
     +--------------------+----------------+-------------+
     | Assembler          |        s       |     Text    |
     +--------------------+----------------+-------------+
     | Object             |        o       |    Binary   |
     +--------------------+----------------+-------------+
     | Archive of objects |        a       |    Binary   |
     +--------------------+----------------+-------------+
     | Precompiled header |       gch      |    Binary   |
     +--------------------+----------------+-------------+
     | Clang AST file     |       ast      |    Binary   |
     +--------------------+----------------+-------------+

.. _clang-bundled-code-object-layout-text:

Bundled Text File Layout
========================

The format of the bundled files is currently very simple: text formats are
concatenated with comments that have a magic string and bundle entry ID in
between.

::

  "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ 1st Bundle Entry ID"
  Bundle 1
  "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ 1st Bundle Entry ID"
  ...
  "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ Nth Bundle Entry ID"
  Bundle N
  "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ 1st Bundle Entry ID"

.. _clang-bundled-code-object-layout:

Bundled Binary File Layout
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
    The target triple of the code object. See `Target Triple
    <https://clang.llvm.org/docs/CrossCompilation.html#target-triple>`_.

    The bundler accepts target triples with or without the optional environment
    field:

    ``<arch><sub>-<vendor>-<sys>``, or
    ``<arch><sub>-<vendor>-<sys>-<env>``

    However, in order to standardize outputs for tools that consume bitcode
    bundles, bundles written by the bundler internally use only the 4-field
    target triple:

    ``<arch><sub>-<vendor>-<sys>-<env>``

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

Archive Unbundling
==================

Unbundling of heterogeneous device archive is done to create device specific
archives. Heterogeneous Device Archive is in a format compatible with GNU ar
utility and contains a collection of bundled device binaries where each bundle
file will contain device binaries for a host and one or more targets. The
output device specific archive is in a format compatible with GNU ar utility
and contains a collection of device binaries for a specific target.

.. code::

  Heterogeneous Device Archive, HDA = {F1.X, F2.X, ..., FN.Y}
  where, Fi = Bundle{Host-DeviceBinary, T1-DeviceBinary, T2-DeviceBinary, ...,
                     Tm-DeviceBinary},
         Ti = {Target i, qualified using Bundle Entry ID},
         X/Y = \*.bc for AMDGPU and \*.cubin for NVPTX

  Device Specific Archive, DSA(Tk) = {F1-Tk-DeviceBinary.X, F2-Tk-DeviceBinary.X, ...
                                      FN-Tk-DeviceBinary.Y}
  where, Fi-Tj-DeviceBinary.X represents device binary of i-th bundled device
  binary file for target Tj.

clang-offload-bundler extracts compatible device binaries for a given target
from the bundled device binaries in a heterogeneous device archive and creates
a target specific device archive without bundling.

clang-offload-bundler determines whether a device binary is compatible with a
target by comparing bundle ID's. Two bundle ID's are considered compatible if:

  * Their offload kind are the same
  * Their target triple are the same
  * Their GPUArch are the same
