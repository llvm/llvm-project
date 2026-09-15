.. _libc_gpu_emissary:

========
Emissary
========

.. note:: This feature is experimental and may change in the future.

Emissary lets GPU device code call **host library** functions -- MPI, HDF5,
``printf``, or an application-supplied API -- by name. Clang lowers a call to
the variadic entry point ``_emissary_exec`` into a packed argument buffer and
a GPU RPC request; a host-side server unpacks the buffer and dispatches to a
handler that invokes the real library.

It builds on the RPC transport described in :ref:`libc_gpu_rpc`, adding two
opcodes (``OFFLOAD_EMISSARY`` and ``OFFLOAD_EMISSARY_DM``) and a runtime
handler registry so that adding support for a new host library is a library
change rather than a compiler release.

Components in this repository
=============================

.. list-table::
   :header-rows: 1

   * - File
     - Role
   * - ``clang/lib/Headers/EmissaryIds.h``
     - Wire ABI: ``_emissary_exec``, ``_PACK_EMIS_IDS``, ``emisArgBuf_t``, API
       id enum, RPC opcodes.
   * - ``clang/lib/CodeGen/CGEmitEmissaryExec.cpp``
     - Packs call-site arguments into the buffer and emits the RPC call.
       Interception lives in ``CGExpr.cpp``.
   * - ``libc/src/__support/RPC/emissary_device_utils.cpp``
     - Device helpers: ``__llvm_emissary_premalloc``,
       ``__llvm_emissary_rpc``, ``__llvm_emissary_rpc_dm``.
   * - ``libc/shared/emissary_rpc_server.h``
     - Host registry (``EmissaryRegister`` / ``EmissaryLookup``), buffer
       unpack, ``EmissaryTop``, ``handleEmissaryOpcodes``.
   * - ``offload/plugins-nextgen/common/src/RPC.cpp``
     - Server thread and opcode routing.
   * - ``libc/test/shared/emissary_registry_test.cpp``
     - Registry unit tests.
