.. libc_uefi_usage:

===================
Using libc for UEFI
===================

.. contents:: Table of Contents
  :depth: 4
  :local:

Using the UEFI C library
========================

Once you have finished building the UEFI C library, it
can be used to run libc or libm functions inside a UEFI
environment. Current, not all C standard functions are
supported in UEFI. Consult the :ref:`list of supported
functions<libc_uefi_support>` for a comprehensive list.

Running a UEFI C library program in QEMU
========================================

QEMU is the preferred way to test programs compiled using
the UEFI C library, it only requires OVMF which is based
on EDKII. It is recommended to create a directory which
serves as a fat32 file system but passed through QEMU.
The following flag is capable of doing that:

.. code-block:: sh

   -drive file=fat:rw:fat32-fs

This will expose the ``fat32-fs`` directory as a fat32
partition. Once QEMU starts, press ESQ a few times to
bring up the EDKII menu. Enter the boot manager and
load the option for the UEFI shell. Typically, EDKII
will expose the fat32 filesystem as ``FS0``. From there,
you can run the following command to run your program.
Here, we are using ``a.out`` as the example since clang
outputs to that filename by default.

.. code-block:: sh

   > FS0:
   FS0:> a.out

