===================
Clang-Doc
===================

.. contents::

.. toctree::
   :maxdepth: 1

:program:`clang-doc` is a tool for generating C and C++ documentation from
source code and comments.

The tool is in a very early development stage, so you might encounter bugs and
crashes. Submitting reports with information about how to reproduce the issue
to `the LLVM bug tracker <https://github.com/llvm/llvm-project/issues/>`_ will definitely help the
project. If you have any ideas or suggestions, please to put a feature request
there.

Use
===

:program:`clang-doc` is a `LibTooling
<https://clang.llvm.org/docs/LibTooling.html>`_-based tool, and so requires a
compile command database for your project (for an example of how to do this
see `How To Setup Tooling For LLVM
<https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html>`_).

By default, the tool will run on all files listed in the given compile commands
database:

.. code-block:: console

  $ clang-doc /path/to/compile_commands.json

The tool can also be used on a single file or multiple files if a build path is
passed with the ``-p`` flag.

.. code-block:: console

  $ clang-doc /path/to/file.cpp -p /path/to/build

Output
======

:program:`clang-doc` produces a directory of documentation. One file is produced
for each namespace and record in the project source code, containing all
documentation (including contained functions, methods, and enums) for that item.

The top-level directory is configurable through the ``output`` flag:

.. code-block:: console

  $ clang-doc -output=output/directory/ compile_commands.json

Configuration
=============

Configuration for :program:`clang-doc` is currently limited to command-line options.
In the future, it may develop the ability to use a configuration file, but no such
efforts are currently in progress.

Options
-------

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

    --doxygen                   - Use only doxygen-style comments to generate docs.
    --extra-arg=<string>        - Additional argument to append to the compiler command line
                                  Can be used several times.
    --extra-arg-before=<string> - Additional argument to prepend to the compiler command line
                                  Can be used several times.
    --format=<value>            - Format for outputted docs.
      =yaml                     -   Documentation in YAML format.
      =md                       -   Documentation in MD format.
      =html                     -   Documentation in HTML format.
    --ignore-map-errors         - Continue if files are not mapped correctly.
    --output=<string>           - Directory for outputting generated files.
    -p=<string>                 - Build path
    --project-name=<string>     - Name of project.
    --public                    - Document only public declarations.
    --repository=<string>       -
                                  URL of repository that hosts code.
                                  Used for links to definition locations.
    --source-root=<string>      -
                                  Directory where processed files are stored.
                                  Links to definition locations will only be
                                  generated if the file is in this dir.
    --stylesheets=<string>      - CSS stylesheets to extend the default styles.

The following flags should only be used if ``format`` is set to ``html``:
- ``repository``
- ``source-root``
- ``stylesheets``
