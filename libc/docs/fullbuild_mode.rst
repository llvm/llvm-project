.. _fullbuild_mode:

==============
Fullbuild Mode
==============

The *fullbuild* mode of LLVM's libc is the mode in which it is to be used as
the only libc (as opposed to the :ref:`overlay_mode` in which it is used along
with the system libc.) In order to use it as the only libc, one will have to
build and install not only the static archives like ``libc.a`` from LLVM's libc,
but also the start-up objects like ``crt1.o`` and the public headers.

The full libc build can be of two types:

.. toctree::
   :maxdepth: 1

   full_host_build
   full_cross_build
