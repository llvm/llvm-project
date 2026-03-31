==============
RISC-V Support
==============

.. contents::
   :local:

Intrinsic Detection Macros
===========================

Clang provides macros to detect which RISC-V intrinsics are supported by the
toolchain.
Note: This is independent from assembler support.

Scalar Intrinsic Detection
---------------------------

Macros of the form ``__riscv_intrinsic_<extension>`` indicate that the
toolchain supports scalar built-in functions for a given extension:

.. code-block:: c

   #if defined(__riscv_intrinsic_zbb)
     // Toolchain supports Zbb intrinsics like __builtin_riscv_orc_b_*
     // These can be used with target attributes
   #endif

Composite extensions are also defined when all their sub-extensions are available, e.g.
 ``__riscv_intrinsic_zkn`` - zbkb + zbkc + zbkx + zkne + zknd + zknh

Vector Intrinsic Detection
---------------------------

Macros of the form ``__riscv_v_intrinsic_<extension>`` indicate that the
toolchain supports vector intrinsics for a given extension:

.. code-block:: c

   #if defined(__riscv_v_intrinsic_zvbb)
     // Toolchain supports vector bit manipulation intrinsics
   #endif

Composite vector crypto extensions are defined when all components are available, e.g.  
 ``__riscv_v_intrinsic_zvkn`` - zvkned + zvknhb + zvkb
