flang - the Fortran compiler
============================

SYNOPSIS
--------

:program:`flang` [*options*] *filename ...*

DESCRIPTION
-----------

:program:`flang` is a Fortran compiler which encompasses preprocessing, parsing,
optimization, code generation, assembly, and linking.  Depending on which
high-level mode setting is passed, Flang will stop before doing a full link.
While Flang is highly integrated, it is important to understand the stages of
compilation, to understand how to invoke it.  These stages are:

Driver
    The flang executable is actually a small driver which controls the overall
    execution of other tools such as the compiler, assembler and linker.
    Typically you do not need to interact with the driver, but you
    transparently use it to run the other tools.

Preprocessing
    This stage handles tokenization of the input source file, macro expansion,
    #include expansion and handling of other preprocessor directives.

Parsing and Semantic Analysis
    This stage parses the input file, translating preprocessor tokens into a
    parse tree.  Once in the form of a parse tree, it applies semantic
    analysis to compute types for expressions as well and determine whether
    the code is well formed. This stage is responsible for generating most of
    the compiler warnings as well as parse errors.

Code Generation and Optimization
    This stage creates the intermediate code (known as "LLVM IR") and ultimately
    to machine code.  This phase is responsible for optimizing the generated
    code and handling target-specific code generation. The output of this stage
    is typically called a ".s" file or "assembly" file.

    Flang also supports the use of an integrated assembler, in which the code
    generator produces object files directly. This avoids the overhead of
    generating the ".s" file and of calling the target assembler.

Assembler
    This stage runs the target assembler to translate the output of the
    compiler into a target object file. The output of this stage is typically
    called a ".o" file or "object" file.

Linker
    This stage runs the target linker to merge multiple object files into an
    executable or dynamic library. The output of this stage is typically called
    an "a.out", ".dylib" or ".so" file.

OPTIONS
-------

.. toctree::
   :maxdepth: 1

   FlangCommandLineOptions
