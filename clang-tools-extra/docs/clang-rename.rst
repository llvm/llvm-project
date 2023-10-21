============
Clang-Rename
============

This is just a test

.. contents::

See also:

.. toctree::
   :maxdepth: 1


:program:`clang-rename` is a C++ refactoring tool. Its purpose is to perform
efficient renaming actions in large-scale projects such as renaming classes,
functions, variables, arguments, namespaces etc.

The tool is in a very early development stage, so you might encounter bugs and
crashes. Submitting reports with information about how to reproduce the issue
to `the LLVM bugtracker <https://bugs.llvm.org>`_ will definitely help the
project. If you have any ideas or suggestions, you might want to put a feature
request there.

Using Clang-Rename
==================

:program:`clang-rename` is a `LibTooling
<https://clang.llvm.org/docs/LibTooling.html>`_-based tool, and it's easier to
work with if you set up a compile command database for your project (for an
example of how to do this see `How To Setup Tooling For LLVM
<https://clang.llvm.org/docs/HowToSetupToolingForLLVM.html>`_). You can also
specify compilation options on the command line after `--`:

.. code-block:: console

  $ clang-rename -offset=42 -new-name=foo test.cpp -- -Imy_project/include -DMY_DEFINES ...


To get an offset of a symbol in a file run

.. code-block:: console

  $ grep -FUbo 'foo' file.cpp


The tool currently supports renaming actions inside a single translation unit
only. It is planned to extend the tool's functionality to support multi-TU
renaming actions in the future.

:program:`clang-rename` also aims to be easily integrated into popular text
editors, such as Vim and Emacs, and improve the workflow of users.

Although a command line interface exists, it is highly recommended to use the
text editor interface instead for better experience.

You can also identify one or more symbols to be renamed by giving the fully
qualified name:

.. code-block:: console

  $ clang-rename -qualified-name=foo -new-name=bar test.cpp

Renaming multiple symbols at once is supported, too. However,
:program:`clang-rename` doesn't accept both `-offset` and `-qualified-name` at
the same time. So, you can either specify multiple `-offset` or
`-qualified-name`.

.. code-block:: console

  $ clang-rename -offset=42 -new-name=bar1 -offset=150 -new-name=bar2 test.cpp

or

.. code-block:: console

  $ clang-rename -qualified-name=foo1 -new-name=bar1 -qualified-name=foo2 -new-name=bar2 test.cpp


Alternatively, {offset | qualified-name} / new-name pairs can be put into a YAML
file:

.. code-block:: yaml

  ---
  - Offset:         42
    NewName:        bar1
  - Offset:         150
    NewName:        bar2
  ...

or

.. code-block:: yaml

  ---
  - QualifiedName:  foo1
    NewName:        bar1
  - QualifiedName:  foo2
    NewName:        bar2
  ...

That way you can avoid spelling out all the names as command line arguments:

.. code-block:: console

  $ clang-rename -input=test.yaml test.cpp

:program:`clang-rename` offers the following options:

.. code-block:: console

  $ clang-rename --help
  USAGE: clang-rename [subcommand] [options] <source0> [... <sourceN>]

  OPTIONS:

  Generic Options:

    -help                      - Display available options (-help-hidden for more)
    -help-list                 - Display list of available options (-help-list-hidden for more)
    -version                   - Display the version of this program

  clang-rename common options:

    -export-fixes=<filename>   - YAML file to store suggested fixes in.
    -extra-arg=<string>        - Additional argument to append to the compiler command line
                                 Can be used several times.
    -extra-arg-before=<string> - Additional argument to prepend to the compiler command line
                                 Can be used several times.
    -force                     - Ignore nonexistent qualified names.
    -i                         - Overwrite edited <file>s.
    -input=<string>            - YAML file to load oldname-newname pairs from.
    -new-name=<string>         - The new name to change the symbol to.
    -offset=<uint>             - Locates the symbol by offset as opposed to <line>:<column>.
    -p <string>                - Build path
    -pl                        - Print the locations affected by renaming to stderr.
    -pn                        - Print the found symbol's name prior to renaming to stderr.
    -qualified-name=<string>   - The fully qualified name of the symbol.

Vim Integration
===============

You can call :program:`clang-rename` directly from Vim! To set up
:program:`clang-rename` integration for Vim see
`clang/tools/clang-rename/clang-rename.py
<https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-rename/clang-rename.py>`_.

Please note that **you have to save all buffers, in which the replacement will
happen before running the tool**.

Once installed, you can point your cursor to symbols you want to rename, press
`<leader>cr` and type new desired name. The `<leader> key
<http://vim.wikia.com/wiki/Mapping_keys_in_Vim_-_Tutorial_(Part_3)#Map_leader>`_
is a reference to a specific key defined by the mapleader variable and is bound
to backslash by default.

Emacs Integration
=================

You can also use :program:`clang-rename` while using Emacs! To set up
:program:`clang-rename` integration for Emacs see
`clang-rename/tool/clang-rename.el
<https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-rename/clang-rename.el>`_.

Once installed, you can point your cursor to symbols you want to rename, press
`M-X`, type `clang-rename` and new desired name.

Please note that **you have to save all buffers, in which the replacement will
happen before running the tool**.
