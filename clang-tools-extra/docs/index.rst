.. title:: Welcome to Extra Clang Tools's documentation!

Introduction
============
Welcome to the clang-tools-extra project which contains extra tools built using
Clang's tooling APIs.

.. toctree::
   :maxdepth: 1

   ReleaseNotes

Contents
========
.. toctree::
   :maxdepth: 2

   clang-tidy/index
   include-fixer
   modularize
   pp-trace
   clang-rename
   clangd/index
   clangd/DeveloperDocumentation
   clang-doc


Doxygen Documentation
=====================
The Doxygen documentation describes the **internal** software that makes up the
tools of clang-tools-extra, not the **external** use of these tools. The Doxygen
documentation contains no instructions about how to use the tools, only the APIs
that make up the software. For usage instructions, please see the user's guide
or reference manual for each tool.

* `Doxygen documentation`_

.. _`Doxygen documentation`: doxygen/annotated.html

.. note::
    This documentation is generated directly from the source code with doxygen.
    Since the tools of clang-tools-extra are constantly under active
    development, what you're about to read is out of date!


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
