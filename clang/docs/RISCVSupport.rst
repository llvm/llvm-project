==============
RISC-V Support
==============

.. contents::
   :local:

Intrinsic Detection Macros
===========================

Clang provides macros to detect which RISC-V intrinsics are supported by the
toolchain. This is only available if intrinsics are ratified, in other word,
experimental intrinsics do not have macro defined.
Note: This is independent from assembler support.

Intrinsic Detection
---------------------------

Macros of the form ``__riscv_intrinsic_<extension>`` indicate that the toolchain
supports intrinsics for a given extension:

.. code-block:: c

  #if defined(__riscv_intrinsic_zbb)
    // Toolchain supports Zbb scalar intrinsics like __riscv_orc_b_*
    // These can be used with target attributes even if -march doesn't include Zbb
    __attribute__((target("arch=+zbb")))
    unsigned long process_with_zbb(unsigned long x) {
      return __riscv_orc_b_64(x);
    }
  #endif

Composite extensions are also defined when all their sub-extensions are available, e.g.
 ``__riscv_intrinsic_zkn`` - zbkb + zbkc + zbkx + zkne + zknd + zknh
