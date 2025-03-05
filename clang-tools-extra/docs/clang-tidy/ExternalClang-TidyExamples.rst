============================
External Clang-Tidy Examples
============================

Introduction
============

This page provides examples of what people have done with :program:`clang-tidy` that 
might serve as useful guides (or starting points) to develop your own checks. 
They may be helpful for necessary things such as how to write the `CMakeLists.txt`
for an out-of-tree plugin of :program:`clang-tidy` checks.

If you know of (or wrote!) a tool or project using :program:`clang-tidy`, please share it 
on `the Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_ for wider visibility and open a 
pull-request on `LLVM Github`_ to have it added here. Since the primary purpose of 
this page is to provide examples that can help developers, the listed projects should
have code available.

As :program:`clang-tidy` is using, for example, the AST Matchers and diagnostics of Clang,
`External Clang Examples`_ may also be useful to look at for such examples.

.. _LLVM Github: https://github.com/llvm/llvm-project
.. _External Clang Examples: https://clang.llvm.org/docs/ExternalClangExamples.html

List of projects and tools
==========================

`<https://github.com/coveooss/clang-tidy-plugin-examples>`_
    "This folder contains :program:`clang-tidy` plugins."
