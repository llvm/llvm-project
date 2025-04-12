.. _libc_uefi_usage:

===================
Using libc for UEFI
===================

.. contents:: Table of Contents
  :depth: 4
  :local:

Using the UEFI C library
========================

Once you have finished :ref:`building<libc_uefi_building>` the UEFI C library
it can be used to run libc or libm functions inside of UEFI Images. Currently,
not all C standard functions are supported in UEFI. Consult the :ref:`list of
supported functions<libc_uefi_support>` for a comprehensive list.
