===========
Usage Modes
===========

The libc can used in two different modes:

#. The **overlay** mode: In this mode, the link order semantics are exploited
   to overlay implementations from LLVM's libc over the system libc. See
   :ref:`overlay_mode` for more information about this mode.
#. The **fullbuild** mode: In this mode, LLVM's libc is used as the only libc
   for the binary. See :ref:`fullbuild_mode` for information about this mode.
