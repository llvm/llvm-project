.. _libc_gpu_rpc:

======================
Remote Procedure Calls
======================

.. contents:: Table of Contents
  :depth: 4
  :local:

Remote Procedure Call Implementation
====================================

Certain features from the standard C library, such as allocation or printing,
require support from the operating system. We instead implement a remote
procedure call (RPC) interface to allow submitting work from the GPU to a host
server that forwards it to the host system.
