===========
Usage Modes
===========

The libc can used in two different modes:

#. The **overlay** mode: In this mode, the link order semantics are exploited
   to overlay implementations from LLVM's libc over the system libc. See
   :ref:`overlay_mode` for more information about this mode. In this mode, libc
   uses the ABI of the system it's being overlayed onto. Headers are NOT
   generated. libllvmlibc.a is the only build artifact.
#. The **fullbuild** mode: In this mode, LLVM's libc is used as the only libc
   for the binary. See :ref:`fullbuild_mode` for information about this mode.
   In this mode, libc uses its own ABI. Headers are generated along with a
   libc.a.
