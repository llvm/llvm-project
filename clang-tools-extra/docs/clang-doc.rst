===================
Clang-Doc
===================

.. contents::

.. toctree::
   :maxdepth: 1

:program:`clang-doc` is a tool for generating C and C++ documenation from 
source code and comments. 

The tool is in a very early development stage, so you might encounter bugs and
crashes. Submitting reports with information about how to reproduce the issue
to `the LLVM bugtracker <https://llvm.org/bugs>`_ will definitely help the
project. If you have any ideas or suggestions, please to put a feature request
there.

Use
=====

:program:`clang-doc` is a `LibTooling
<https://clang.llvm.org/docs/LibTooling.html>`_-based tool, and so requires a
compile command database for your project (for an example of how to do this 
see `How To Setup Tooling For LLVM
<https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html>`_).

The tool can be used on a single file or multiple files as defined in 
the compile commands database:

.. code-block:: console

  $ clang-doc /path/to/file.cpp -p /path/to/compile/commands

This generates an intermediate representation of the declarations and their
associated information in the specified TUs, serialized to LLVM bitcode.

As currently implemented, the tool is only able to parse TUs that can be 
stored in-memory. Future additions will extend the current framework to use
map-reduce frameworks to allow for use with large codebases.

:program:`clang-doc` offers the following options:

.. code-block:: console

	$ clang-doc --help
  USAGE: clang-doc [options] <source0> [... <sourceN>]

  OPTIONS:

  Generic Options:

    -help                      - Display available options (-help-hidden for more)
    -help-list                 - Display list of available options (-help-list-hidden for more)
    -version                   - Display the version of this program

  clang-doc options:

    -doxygen                   - Use only doxygen-style comments to generate docs.
    -dump                      - Dump intermediate results to bitcode file.
    -extra-arg=<string>        - Additional argument to append to the compiler command line
    -extra-arg-before=<string> - Additional argument to prepend to the compiler command line
    -omit-filenames            - Omit filenames in output.
    -output=<string>           - Directory for outputting generated files.
    -p=<string>                - Build path
