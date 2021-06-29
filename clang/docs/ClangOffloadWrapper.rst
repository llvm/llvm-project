=====================
Clang Offload Wrapper
=====================

.. contents::
   :local:

.. _clang-offload-wrapper:

Introduction
============

This tool is used in OpenMP offloading toolchain to embed device code objects
(usually ELF) into a wrapper host llvm IR (bitcode) file. The wrapper host IR
is then assembled and linked with host code objects to generate the executable
binary. See :ref:`multi-image-binary-embedding-execution` for more details.

Usage
=====

This tool can be used as follows:

.. code-block:: console

  $ clang-offload-wrapper -help
  OVERVIEW: A tool to create a wrapper bitcode for offload target binaries.
  Takes offload target binaries as input and produces bitcode file containing
  target binaries packaged as data and initialization code which registers
  target binaries in offload runtime.

  USAGE: clang-offload-wrapper [options] <input files>

  OPTIONS:

  Generic Options:

    --help                             - Display available options (--help-hidden for more)
    --help-list                        - Display list of available options (--help-list-hidden for more)
    --version                          - Display the version of this program

  clang-offload-wrapper options:

    -o=<filename>                      - Output filename
    --offload-arch=<offload-arch-name> - Contains offload-arch of the following target binary.
    --target=<triple>                  - Target triple for the output amdgpu-amdhsa-memory-model-code-sequences-gfx10-table

Example
=======

.. code-block:: console

  clang-offload-wrapper -target host-triple -o host-wrapper.bc --offload-arch=gfx906 gfx906-binary.out --offload-arch=gfx90a gfx90a-binary.out


.. _openmp-device-binary_embedding:

OpenMP Device Binary Embedding
==============================

Various structures and functions used in the wrapper host IR form the interface
between the executable binary and the OpenMP runtime.

Enum Types
----------

:ref:`table-offloading-declare-target-flags` lists different flag for
offloading entries.

  .. table:: Offloading Declare Target Flags Enum
    :name: table-offloading-declare-target-flags

    +-------------------------+-------+------------------------------------------------------------------+
    |          Name           | Value | Description                                                      |
    +=========================+=======+==================================================================+
    | OMP_DECLARE_TARGET_LINK | 0x01  | Mark the entry as having a 'link' attribute (w.r.t. link clause) |
    +-------------------------+-------+------------------------------------------------------------------+
    | OMP_DECLARE_TARGET_CTOR | 0x02  | Mark the entry as being a global constructor                     |
    +-------------------------+-------+------------------------------------------------------------------+
    | OMP_DECLARE_TARGET_DTOR | 0x04  | Mark the entry as being a global destructor                      |
    +-------------------------+-------+------------------------------------------------------------------+


Structure Types
---------------

:ref:`table-tgt_offload_entry`, :ref:`table-tgt_device_image`,
:ref:`table-tgt_bin_desc`, and :ref:`table-tgt_image_info` are the structures
used in the wrapper host IR.

  .. table:: __tgt_offload_entry structure
    :name: table-tgt_offload_entry

    +---------+------------+------------------------------------------------------------------------------------+
    |   Type  | Identifier | Description                                                                        |
    +=========+============+====================================================================================+
    |  void*  |    addr    | Address of global symbol within device image (function or global)                  |
    +---------+------------+------------------------------------------------------------------------------------+
    |  char*  |    name    | Name of the symbol                                                                 |
    +---------+------------+------------------------------------------------------------------------------------+
    |  size_t |    size    | Size of the entry info (0 if it is a function)                                     |
    +---------+------------+------------------------------------------------------------------------------------+
    | int32_t |    flags   | Flags associated with the entry (see :ref:`table-offloading-declare-target-flags`) |
    +---------+------------+------------------------------------------------------------------------------------+
    | int32_t |  reserved  | Reserved, to be used by the runtime library.                                       |
    +---------+------------+------------------------------------------------------------------------------------+

  .. table:: __tgt_device_image structure
    :name: table-tgt_device_image

    +----------------------+--------------+----------------------------------------+
    |         Type         |  Identifier  | Description                            |
    +======================+==============+========================================+
    |         void*        |  ImageStart  | Pointer to the target code start       |
    +----------------------+--------------+----------------------------------------+
    |         void*        |   ImageEnd   | Pointer to the target code end         |
    +----------------------+--------------+----------------------------------------+
    | __tgt_offload_entry* | EntriesBegin | Begin of table with all target entries |
    +----------------------+--------------+----------------------------------------+
    | __tgt_offload_entry* |  EntriesEnd  | End of table (non inclusive)           |
    +----------------------+--------------+----------------------------------------+

  .. table:: __tgt_bin_desc structure
    :name: table-tgt_bin_desc

    +----------------------+------------------+------------------------------------------+
    |         Type         |    Identifier    | Description                              |
    +======================+==================+==========================================+
    |        int32_t       |  NumDeviceImages | Number of device types supported         |
    +----------------------+------------------+------------------------------------------+
    |  __tgt_device_image* |   DeviceImages   | Array of device images (1 per dev. type) |
    +----------------------+------------------+------------------------------------------+
    | __tgt_offload_entry* | HostEntriesBegin | Begin of table with all host entries     |
    +----------------------+------------------+------------------------------------------+
    | __tgt_offload_entry* |  HostEntriesEnd  | End of table (non inclusive)             |
    +----------------------+------------------+------------------------------------------+

  .. table:: __tgt_image_info structure
    :name: table-tgt_image_info

    +---------+---------------+-----------------------------------------------+
    |   Type  |   Identifier  | Description                                   |
    +=========+===============+===============================================+
    | int32_t |    version    | The version of this struct                    |
    +---------+---------------+-----------------------------------------------+
    | int32_t |  image_number | Image number in image library starting from 0 |
    +---------+---------------+-----------------------------------------------+
    | int32_t | number_images | Number of images, used for initial allocation |
    +---------+---------------+-----------------------------------------------+
    |  char*  |  offload_arch | Target ID for which this image was compiled   |
    +---------+---------------+-----------------------------------------------+
    |  char*  | compile_opts  | reserved for future use                       |
    +---------+---------------+-----------------------------------------------+

Global Variables
----------------

:ref:`table-global-variables` lists various global variables, along with their
type and their explicit ELF sections, which are used to store device images and
related symbols.

  .. table:: Global Variables
    :name: table-global-variables

    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    |            Variable            |         Type        |       ELF Section       |                    Description                    |
    +================================+=====================+=========================+===================================================+
    | __start_omp_offloading_entries | __tgt_offload_entry | .omp_offloading_entries | Begin symbol for the offload entries table.       |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | __stop_omp_offloading_entries  | __tgt_offload_entry | .omp_offloading_entries | End symbol for the offload entries table.         |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | __dummy.omp_offloading.entry   | __tgt_offload_entry | .omp_offloading_entries | Dummy zero-sized object in the offload entries    |
    |                                |                     |                         | section to force linker to define begin/end       |
    |                                |                     |                         | symbols defined above.                            |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .omp_offloading.device_image   |  __tgt_device_image | .omp_offloading_entries | ELF device code object of the first image.        |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .omp_offloading.device_image.N |  __tgt_device_image | .omp_offloading_entries | ELF device code object of the (N+1)th image.      |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .omp_offloading.device_images  |  __tgt_device_image | .omp_offloading_entries | Array of images.                                  |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .omp_offloading.descriptor     | __tgt_bin_desc      | .omp_offloading_entries | Binary descriptor object (see details below).     |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | __offload_arch                 | string              | .offload_arch_list      | Target ID string of the first image.              |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .offload_image_info            | __tgt_image_info    | .omp_offloading_entries | Object containing target ID of the first image.   |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | __offload_arch.N               | string              | .offload_arch_list      | Target ID string of the (N+1)th image.            |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+
    | .offload_image_info.N          | __tgt_image_info    | .omp_offloading_entries | Object containing target ID of the (N+1)th image. |
    +--------------------------------+---------------------+-------------------------+---------------------------------------------------+

Binary Descriptor for Device Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This object is passed to the offloading runtime at program startup and it
describes all device images available in the executable or shared library. It
is defined as follows:

.. code-block:: console

  __attribute__((visibility("hidden")))
  extern __tgt_offload_entry *__start_omp_offloading_entries;
  __attribute__((visibility("hidden")))
  extern __tgt_offload_entry *__stop_omp_offloading_entries;

  static const char Image0[] = { <Bufs.front() contents> };
  ...
  static const char ImageN[] = { <Bufs.back() contents> };

  static const __tgt_device_image Images[] = {
    {
      Image0,                            /*ImageStart*/
      Image0 + sizeof(Image0),           /*ImageEnd*/
      __start_omp_offloading_entries,    /*EntriesBegin*/
      __stop_omp_offloading_entries      /*EntriesEnd*/
    },
    ...
    {
      ImageN,                            /*ImageStart*/
      ImageN + sizeof(ImageN),           /*ImageEnd*/
      __start_omp_offloading_entries,    /*EntriesBegin*/
      __stop_omp_offloading_entries      /*EntriesEnd*/
    }
  };

  static const __tgt_bin_desc BinDesc = {
    sizeof(Images) / sizeof(Images[0]),  /*NumDeviceImages*/
    Images,                              /*DeviceImages*/
    __start_omp_offloading_entries,      /*HostEntriesBegin*/
    __stop_omp_offloading_entries        /*HostEntriesEnd*/
  };

Global Constructor and Destructor
---------------------------------

Global constructor (``.omp_offloading.descriptor_reg()``) registers the library
of images with the runtime by calling ``__tgt_register_lib()`` function. The
cunstructor is explicitly defined in ``.text.startup`` section. It calls
``__tgt_register_image_info()`` function for each ``.offload_image_info.N``
before calling registration function. Similarly, global destructor
(``.omp_offloading.descriptor_unreg()``) calls ``__tgt_unregister_lib()`` for
the unregistration and is also defined in ``.text.startup`` section.

.. _multi-image-binary-embedding-execution:

Multi-image Binary Embedding and Execution for OpenMP
=====================================================
For each offloading target, device ELF code objects are generated by ``clang``,
``opt``, ``llc``, and ``lld`` pipeline. These code objects along with the
target id of the offloading target devices are passed to the
``clang-offload-wrapper``.

  * At compile time, the ``clang-offload-wrapper`` tool takes the following
    actions:

    * It embeds the ELF code objects for the device into the host code (see
      :ref:`openmp-device-binary_embedding`).
    * It creates internal labels to these embedded device code objects
      (``.offload_image_info.N``).
    * It creates a global constructor to get the address of the embedded device
      code through ``.offload_image_info.N`` structure and to register the
      device code.
    * It also creates a new ELF section ``.offload_arch_list`` with an array of
      null-terminated strings where each string (``__offload_arch.N``) provides
      the target ID of an image.

  * At execution time:

    * The global constructor gets run and it registers the device image.
    * The runtime looks for an image that is compatible with the offload
      environment. It uses the ``offload-arch`` library to obtain underlying
      system's environment. It's the target ID for AMDGPU and the processor
      name for other offloading targets.
