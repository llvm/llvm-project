=======================
External Clang-Tidy Examples
=======================

Introduction
============

This page provides examples of what people have done with `clang-tidy`` that 
might serve as useful guides (or starting points) to develop your own checks. 
They may be helpful for necessary things such as how to write CMakeLists.txt 
for an out-of-tree plugin of `clang-tidy` checks.

If you know of (or wrote!) a tool or project using clang-tidy, please share it 
on `the Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_ for wider visibility and open a 
pull-request on `LLVM Github`_ to have it added here. Since the primary purpose of 
this page is to provide examples that can help developers, generally they must have 
code available.

As `clang-tidy` shares C++ AST Matchers with Clang diagnostics, `External Clang Examples`_ 
may also be useful to look at for examples.

.. _LLVM Github: https://github.com/llvm/llvm-project
.. _External Clang Examples: https://clang.llvm.org/docs/ExternalClangExamples.html

https://clang.llvm.org/docs/ExternalClangExamples.html

List of projects and tools
==========================

`<https://github.com/coveooss/clang-tidy-plugin-examples/tree/main>`_
    "This folder contains clang-tidy plugins."