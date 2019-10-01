==================================
Developer documentation for clangd
==================================

.. toctree::
   :maxdepth: 1

   Extensions

Compiling clangd
================

To build clangd from source, please follow the instructions for `building Clang
<https://clang.llvm.org/get_started.html>`_ and include LLVM, Clang, and the
"extra Clang tools" in your build.

Contributing to clangd
======================

A good place for interested contributors is the `Clangd developer mailing list
<https://lists.llvm.org/mailman/listinfo/clangd-dev>`_. For discussions with
the broader community on topics not only related to Clangd, use `Clang
developer mailing list <https://lists.llvm.org/mailman/listinfo/cfe-dev>`_.  If
you're also interested in contributing patches to clangd, take a look at the
`LLVM Developer Policy <https://llvm.org/docs/DeveloperPolicy.html>`_ and `Code
Reviews <https://llvm.org/docs/Phabricator.html>`_ page. Contributions of new
features to the `Language Server Protocol
<https://github.com/Microsoft/language-server-protocol>`_ itself would also be
very useful, so that clangd can eventually implement them in a conforming way.
