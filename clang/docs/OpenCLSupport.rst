.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

.. contents::
   :local:

==================
OpenCL Support
==================

Clang fully supports all OpenCL C versions from 1.1 to 2.0.

Please refer to `Bugzilla
<https://bugs.llvm.org/buglist.cgi?component=OpenCL&list_id=172679&product=clang&resolution=--->`_
for the most up to date bug reports.


C++ for OpenCL Implementation Status
====================================

Bugzilla bugs for this functionality are typically prefixed
with '[C++]'.

Differences to OpenCL C
-----------------------

TODO!

Missing features or with limited support
----------------------------------------

- Use of ObjC blocks is disabled.

- Global destructor invocation is not generated correctly.

- Initialization of objects in `__constant` address spaces is not guaranteed to work.

- `addrspace_cast` operator is not supported.
