# Welcome to Flang's documentation

Flang is LLVM's Fortran frontend that can be found
[here](https://github.com/llvm/llvm-project/tree/main/flang). It is often
referred to as "LLVM Flang" to differentiate itself from ["Classic
Flang"](https://github.com/flang-compiler/flang) - these are two separate and
independent Fortran compilers. LLVM Flang is under active development. While it
is capable of generating executables for a number of examples, some
functionality is still missing. See [GettingInvolved](GettingInvolved) for tips
on how to get in touch with us and to learn more about the current status.

```eval_rst
.. toctree::
   :titlesonly:

   ReleaseNotes
```

# Contributing to Flang

```eval_rst
.. toctree::
   :titlesonly:

   C++17
   C++style
   FortranForCProgrammers
   GettingInvolved
   ImplementingASemanticCheck
   PullRequestChecklist
```

# Design Documents

```eval_rst
.. toctree::
   :titlesonly:

   ArrayComposition
   BijectiveInternalNameUniquing
   Calls
   Character
   ControlFlowGraph
   Directives
   DoConcurrent
   Extensions
   FIRLangRef
   FlangDriver
   FortranIR
   FortranLLVMTestSuite
   IORuntimeInternals
   Intrinsics
   LabelResolution
   ModFiles
   OpenMP-4.5-grammar.md
   OpenMP-semantics
   OptionComparison
   Overview
   ParserCombinators
   Parsing
   Preprocessing
   RuntimeDescriptor
   RuntimeTypeInfo
   Semantics
   f2018-grammar.md
```

# Indices and tables

```eval_rst
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
