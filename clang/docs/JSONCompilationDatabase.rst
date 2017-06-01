==============================================
JSON Compilation Database Format Specification
==============================================

This document describes a format for specifying how to replay single
compilations independently of the build system.

Background
==========

Tools based on the C++ Abstract Syntax Tree need full information how to
parse a translation unit. Usually this information is implicitly
available in the build system, but running tools as part of the build
system is not necessarily the best solution:

-  Build systems are inherently change driven, so running multiple tools
   over the same code base without changing the code does not fit into
   the architecture of many build systems.
-  Figuring out whether things have changed is often an IO bound
   process; this makes it hard to build low latency end user tools based
   on the build system.
-  Build systems are inherently sequential in the build graph, for
   example due to generated source code. While tools that run
   independently of the build still need the generated source code to
   exist, running tools multiple times over unchanging source does not
   require serialization of the runs according to the build dependency
   graph.

Supported Systems
=================

Currently `CMake <http://cmake.org>`_ (since 2.8.5) supports generation
of compilation databases for Unix Makefile builds (Ninja builds in the
works) with the option ``CMAKE_EXPORT_COMPILE_COMMANDS``.

For projects on Linux, there is an alternative to intercept compiler
calls with a tool called `Bear <https://github.com/rizsotto/Bear>`_.

Clang's tooling interface supports reading compilation databases; see
the :doc:`LibTooling documentation <LibTooling>`. libclang and its
python bindings also support this (since clang 3.2); see
`CXCompilationDatabase.h </doxygen/group__COMPILATIONDB.html>`_.

Format
======

A compilation database is a JSON file, which consist of an array of
"command objects", where each command object specifies one way a
translation unit is compiled in the project.

Each command object contains the translation unit's main file, the
working directory of the compile run and the actual compile command.

Example:

::

    [
      { "directory": "/home/user/llvm/build",
        "command": "/usr/bin/clang++ -Irelative -DSOMEDEF=\"With spaces, quotes and \\-es.\" -c -o file.o file.cc",
        "file": "file.cc" },
      ...
    ]

The contracts for each field in the command object are:

-  **directory:** The working directory of the compilation. All paths
   specified in the **command** or **file** fields must be either
   absolute or relative to this directory.
-  **file:** The main translation unit source processed by this
   compilation step. This is used by tools as the key into the
   compilation database. There can be multiple command objects for the
   same file, for example if the same source file is compiled with
   different configurations.
-  **command:** The compile command executed. After JSON unescaping,
   this must be a valid command to rerun the exact compilation step for
   the translation unit in the environment the build system uses.
   Parameters use shell quoting and shell escaping of quotes, with '``"``'
   and '``\``' being the only special characters. Shell expansion is not
   supported.
-  **arguments:** The compile command executed as list of strings.
   Either **arguments** or **command** is required.
-  **output:** The name of the output created by this compilation step.
   This field is optional. It can be used to distinguish different processing
   modes of the same input file.

Build System Integration
========================

The convention is to name the file compile\_commands.json and put it at
the top of the build directory. Clang tools are pointed to the top of
the build directory to detect the file and use the compilation database
to parse C++ code in the source tree.
