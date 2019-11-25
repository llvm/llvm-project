Generating Public and Internal headers
======================================

Other libc implementations make use of preprocessor macro tricks to make header
files platform agnostic. When macros aren't suitable, they rely on build
system tricks to pick the right set of files to compile and export. While these
approaches have served them well, parts of their systems have become extremely
complicated making it hard to modify, extend or maintain. To avoid these
problems in llvm-libc, we use a header generation mechanism. The mechanism is
driven by a *header configuration language*.

Header Configuration Language
-----------------------------

Header configuration language consists of few special *commands*. The header
generation mechanism takes a an input file, which has an extension of
``.h.def``, and produces a header file with ``.h`` extension. The header
configuration language commands are listed in the input ``.h.def`` file. While
reading a ``.h.def`` file, the header generation tool does two things:

1. Copy the lines not containing commands as is into the output ``.h`` file.
2. Replace the line on which a command occurs with some other text as directed
   by the command. The replacment text can span multiple lines.

Command syntax
~~~~~~~~~~~~~~

A command should be listed on a line by itself, and should not span more than
one line. The first token to appear on the line is the command name prefixed
with ``%%``. For example, a line with the ``include_file`` command should start
with ``%%include_file``. There can be indentation spaces before the ``%%``
prefix.

Most commands typically take arguments. They are listed as a comma separated
list of named identifiers within parenthesis, similar to the C function call
syntax. Before performing the action corresponding to the command, the header
generator replaces the arguments with concrete values.

Argument Syntax
~~~~~~~~~~~~~~~

Arguments are named indentifiers but prefixed with ``$`` and enclosed in ``{``
and ``}``. For example, ``${path_to_constants}``.

Comments
~~~~~~~~

There can be cases wherein one wants to add comments in the .h.def file but
does not want them to be copied into the generated header file. Such comments
can be added by beginning the comment lines with the ``<!>`` prefix. Currently,
comments have to be on lines of their own. That is, they cannot be suffixes like
this:

```
%%include_file(a/b/c) <!> Path to c in b of a.  !!! WRONG SYNTAX
```

Available Commands
------------------

Sub-sections below describe the commands currently available. Under each command
is the discription of the arugments to the command, and the action taken by the
header generation tool when processing a command.

``include_file``
~~~~~~~~~~~~~~~~

This is a replacement command which should be listed in an input ``.h.def``
file.

Arguments

  * **path argument** - An argument representing a path to a file. The file
    should have an extension of ``.h.inc``.

Action

  This command instructs that the line on which the command appears should be
  replaced by the contents of the file whose path is passed as argument to the
  command.

``begin``
~~~~~~~~~

This is not a replacement command. It is an error to list it in the input
``.h.def`` file. It is normally listed in the files included by the
``include_file`` command (the ``.h.inc`` files). A common use of this command it
mark the beginning of what is to be included. This prevents copying items like
license headers into the generated header file.

Arguments

  None.

Action

  The header generator will only include content starting from the line after the
  line on which this command is listed.

``public_api``
~~~~~~~~~~~~~~

This is a replacement command which should be listed in an input ``.h.def``
file. The header file generator will replace this command with the public API of
the target platform. See the build system document for more information on the
relevant build rules. Also, see "Mechanics of public_api" to learn the mechanics
of how the header generator replaces this command with the public API.

Arguments

  None.

Action

  The header generator will replace this command with the public API to be exposed
  from the generated header file.
