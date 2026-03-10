# Welcome to Flang's documentation

Flang is LLVM's Fortran frontend that can be found
[here](https://github.com/llvm/llvm-project/tree/main/flang). It is often
referred to as "LLVM Flang" to differentiate itself from ["Classic
Flang"](https://github.com/flang-compiler/flang) - these are two separate and
independent Fortran compilers. LLVM Flang is under active development. While it
is capable of generating executables for a number of examples, some
functionality is still missing. See [Getting Involved](GettingInvolved.md) for tips
on how to get in touch with us and to learn more about the current status.

```{eval-rst}
.. toctree::
   :titlesonly:

   ReleaseNotes
```

# Using Flang

```{eval-rst}
.. toctree::
   :titlesonly:

   GettingStarted
   FlangCommandLineReference
   FortranStandardsSupport
   Extensions
   Directives
   OpenMPSupport
   Real16MathSupport
   Unsigned
   FAQ
```

# Contributing to Flang

```{eval-rst}
.. toctree::
   :titlesonly:

   C++17
   C++style
   DesignGuideline
   FortranForCProgrammers
   GettingInvolved
   ImplementingASemanticCheck
   PullRequestChecklist
```

# Design Documents

```{eval-rst}
.. toctree::
   :titlesonly:

   Aliasing
   AliasingAnalysisFIR
   ArrayComposition
   ArrayRepacking
   AssumedRank
   BijectiveInternalNameUniquing
   Calls
   Character
   ComplexOperations
   ControlFlowGraph
   DebugGeneration
   DoConcurrent
   DoConcurrentConversionToOpenMP
   F202X
   FIRArrayOperations
   FIRLangRef
   FlangDriver
   FortranFeatureHistory
   FortranIR
   FortranLLVMTestSuite
   HighLevelFIR
   IORuntimeInternals
   InternalProcedureTrampolines
   Intrinsics
   IntrinsicTypes
   LabelResolution
   ModFiles
   OpenACC
   OpenACC-descriptor-management.md
   OpenMP-4.5-grammar.md
   OpenMP-declare-target
   OpenMP-descriptor-management
   OpenMP-semantics
   OptionComparison
   Overview
   ParallelMultiImageFortranRuntime
   ParameterizedDerivedTypes
   ParserCombinators
   Parsing
   PolymorphicEntities
   Preprocessing
   ProcedurePointer
   RuntimeDescriptor
   RuntimeEnvironment
   RuntimeTypeInfo
   Semantics
   f2018-grammar.md
   fstack-arrays
```

# Indices and tables

```{eval-rst}
* :ref:`genindex`
* :ref:`search`
```
