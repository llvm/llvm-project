.. _index:

lld - The LLVM Linker
=====================

lld contains two linkers whose architectures are different from each other.

.. toctree::
   :maxdepth: 1

   NewLLD
   AtomLLD

Source
------

lld is available in the LLVM SVN repository::

  svn co http://llvm.org/svn/llvm-project/lld/trunk lld

lld is also available via the read-only git mirror::

  git clone http://llvm.org/git/lld.git

Put it in llvm's tools/ directory, rerun cmake, then build target lld.
