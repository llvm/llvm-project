.. raw:: html

      <style type="text/css">
        .versionbadge { background-color: #1c913d; height: 20px; display: inline-block; min-width: 120px; text-align: center; border-radius: 5px; color: #FFFFFF; font-family: "Verdana,Geneva,DejaVu Sans,sans-serif"; }
      </style>

.. role:: versionbadge

==========================
Clang-Format Style Options
==========================

:doc:`ClangFormatStyleOptions` describes configurable formatting style options
supported by :doc:`LibFormat` and :doc:`ClangFormat`.

When using :program:`clang-format` command line utility or
``clang::format::reformat(...)`` functions from code, one can either use one of
the predefined styles (LLVM, Google, Chromium, Mozilla, WebKit, Microsoft) or
create a custom style by configuring specific style options.


Configuring Style with clang-format
===================================

:program:`clang-format` supports two ways to provide custom style options:
directly specify style configuration in the ``-style=`` command line option or
use ``-style=file`` and put style configuration in the ``.clang-format`` or
``_clang-format`` file in the project directory.

When using ``-style=file``, :program:`clang-format` for each input file will
try to find the ``.clang-format`` file located in the closest parent directory
of the input file. When the standard input is used, the search is started from
the current directory.

When using ``-style=file:<format_file_path>``, :program:`clang-format` for
each input file will use the format file located at `<format_file_path>`.
The path may be absolute or relative to the working directory.

The ``.clang-format`` file uses YAML format:

.. code-block:: yaml

  key1: value1
  key2: value2
  # A comment.
  ...

The configuration file can consist of several sections each having different
``Language:`` parameter denoting the programming language this section of the
configuration is targeted at. See the description of the **Language** option
below for the list of supported languages. The first section may have no
language set, it will set the default style options for all languages.
Configuration sections for specific language will override options set in the
default section.

When :program:`clang-format` formats a file, it auto-detects the language using
the file name. When formatting standard input or a file that doesn't have the
extension corresponding to its language, ``-assume-filename=`` option can be
used to override the file name :program:`clang-format` uses to detect the
language.

An example of a configuration file for multiple languages:

.. code-block:: yaml

  ---
  # We'll use defaults from the LLVM style, but with 4 columns indentation.
  BasedOnStyle: LLVM
  IndentWidth: 4
  ---
  Language: Cpp
  # Force pointers to the type for C++.
  DerivePointerAlignment: false
  PointerAlignment: Left
  ---
  Language: JavaScript
  # Use 100 columns for JS.
  ColumnLimit: 100
  ---
  Language: Proto
  # Don't format .proto files.
  DisableFormat: true
  ---
  Language: CSharp
  # Use 100 columns for C#.
  ColumnLimit: 100
  ...

An easy way to get a valid ``.clang-format`` file containing all configuration
options of a certain predefined style is:

.. code-block:: console

  clang-format -style=llvm -dump-config > .clang-format

When specifying configuration in the ``-style=`` option, the same configuration
is applied for all input files. The format of the configuration is:

.. code-block:: console

  -style='{key1: value1, key2: value2, ...}'


Disabling Formatting on a Piece of Code
=======================================

Clang-format understands also special comments that switch formatting in a
delimited range. The code between a comment ``// clang-format off`` or
``/* clang-format off */`` up to a comment ``// clang-format on`` or
``/* clang-format on */`` will not be formatted. The comments themselves will be
formatted (aligned) normally. Also, a colon (``:``) and additional text may
follow ``// clang-format off`` or ``// clang-format on`` to explain why
clang-format is turned off or back on.

.. code-block:: c++

  int formatted_code;
  // clang-format off
      void    unformatted_code  ;
  // clang-format on
  void formatted_code_again;


Configuring Style in Code
=========================

When using ``clang::format::reformat(...)`` functions, the format is specified
by supplying the `clang::format::FormatStyle
<https://clang.llvm.org/doxygen/structclang_1_1format_1_1FormatStyle.html>`_
structure.


Configurable Format Style Options
=================================

This section lists the supported style options. Value type is specified for
each option. For enumeration types possible values are specified both as a C++
enumeration member (with a prefix, e.g. ``LS_Auto``), and as a value usable in
the configuration (without a prefix: ``Auto``).

.. _BasedOnStyle:

**BasedOnStyle** (``String``) :ref:`¶ <BasedOnStyle>`
  The style used for all options not specifically set in the configuration.

  This option is supported only in the :program:`clang-format` configuration
  (both within ``-style='{...}'`` and the ``.clang-format`` file).

  Possible values:

  * ``LLVM``
    A style complying with the `LLVM coding standards
    <https://llvm.org/docs/CodingStandards.html>`_
  * ``Google``
    A style complying with `Google's C++ style guide
    <https://google.github.io/styleguide/cppguide.html>`_
  * ``Chromium``
    A style complying with `Chromium's style guide
    <https://chromium.googlesource.com/chromium/src/+/refs/heads/main/styleguide/styleguide.md>`_
  * ``Mozilla``
    A style complying with `Mozilla's style guide
    <https://firefox-source-docs.mozilla.org/code-quality/coding-style/index.html>`_
  * ``WebKit``
    A style complying with `WebKit's style guide
    <https://www.webkit.org/coding/coding-style.html>`_
  * ``Microsoft``
    A style complying with `Microsoft's style guide
    <https://docs.microsoft.com/en-us/visualstudio/ide/editorconfig-code-style-settings-reference>`_
  * ``GNU``
    A style complying with the `GNU coding standards
    <https://www.gnu.org/prep/standards/standards.html>`_
  * ``InheritParentConfig``
    Not a real style, but allows to use the ``.clang-format`` file from the
    parent directory (or its parent if there is none). If there is no parent
    file found it falls back to the ``fallback`` style, and applies the changes
    to that.

    With this option you can overwrite some parts of your main style for your
    subdirectories. This is also possible through the command line, e.g.:
    ``--style={BasedOnStyle: InheritParentConfig, ColumnLimit: 20}``

.. START_FORMAT_STYLE_OPTIONS
..
  This section of the file is automatically generated by the
  dump_format_style.py tool and should be updated manually.
.. END_FORMAT_STYLE_OPTIONS

Adding additional style options
===============================

Each additional style option adds costs to the clang-format project. Some of
these costs affect the clang-format development itself, as we need to make
sure that any given combination of options work and that new features don't
break any of the existing options in any way. There are also costs for end users
as options become less discoverable and people have to think about and make a
decision on options they don't really care about.

The goal of the clang-format project is more on the side of supporting a
limited set of styles really well as opposed to supporting every single style
used by a codebase somewhere in the wild. Of course, we do want to support all
major projects and thus have established the following bar for adding style
options. Each new style option must:

  * be used in a project of significant size (have dozens of contributors)
  * have a publicly accessible style guide
  * have a person willing to contribute and maintain patches

Examples
========

A style similar to the `Linux Kernel style
<https://www.kernel.org/doc/html/latest/process/coding-style.html>`_:

.. code-block:: yaml

  BasedOnStyle: LLVM
  IndentWidth: 8
  UseTab: Always
  BreakBeforeBraces: Linux
  AllowShortIfStatementsOnASingleLine: false
  IndentCaseLabels: false

The result is (imagine that tabs are used for indentation here):

.. code-block:: c++

  void test()
  {
          switch (x) {
          case 0:
          case 1:
                  do_something();
                  break;
          case 2:
                  do_something_else();
                  break;
          default:
                  break;
          }
          if (condition)
                  do_something_completely_different();

          if (x == y) {
                  q();
          } else if (x > y) {
                  w();
          } else {
                  r();
          }
  }

A style similar to the default Visual Studio formatting style:

.. code-block:: yaml

  UseTab: Never
  IndentWidth: 4
  BreakBeforeBraces: Allman
  AllowShortIfStatementsOnASingleLine: false
  IndentCaseLabels: false
  ColumnLimit: 0

The result is:

.. code-block:: c++

  void test()
  {
      switch (suffix)
      {
      case 0:
      case 1:
          do_something();
          break;
      case 2:
          do_something_else();
          break;
      default:
          break;
      }
      if (condition)
          do_something_completely_different();

      if (x == y)
      {
          q();
      }
      else if (x > y)
      {
          w();
      }
      else
      {
          r();
      }
  }
