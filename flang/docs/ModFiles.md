<!--===- docs/ModFiles.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Module Files

```{contents}
---
local:
---
```

Module files hold information from a module that is necessary to compile 
program units that depend on the module.

## Name

Module files must be searchable by module name. They are typically named
`<modulename>.mod`. The advantage of using `.mod` is that it is consistent with
other compilers so users will know what they are. Also, makefiles and scripts
often use `rm *.mod` to clean up.

The disadvantage of using the same name as other compilers is that it is not
clear which compiler created a `.mod` file and files from multiple compilers
cannot be in the same directory. This could be solved by adding something
between the module name and extension, e.g. `<modulename>-f18.mod`.

## Format

Module files will be Fortran source.
Declarations of all visible entities will be included, along with private
entities that they depend on.
Entity declarations that span multiple statements will be collapsed into
a single *type-declaration-statement*.
Executable statements will be omitted.

### Header

There will be a header containing extra information that cannot be expressed
in Fortran. This will take the form of a comment or directive
at the beginning of the file.

If it's a comment, the module file reader would have to strip it out and
perform *ad hoc* parsing on it. If it's a directive the compiler could
parse it like other directives as part of the grammar.
Processing the header before parsing might result in better error messages
when the `.mod` file is invalid.

Regardless of whether the header is a comment or directive we can use the
same string to introduce it: `!mod$`.

Information in the header:
- Magic string to confirm it is an f18 `.mod` file
- Version information: to indicate the version of the file format, in case it changes,
  and the version of the compiler that wrote the file, for diagnostics.
- Checksum of the body of the current file
- Modules we depend on and the checksum of their module file when the current
  module file is created
- The source file that produced the `.mod` file? This could be used in error messages.

### Body

The body will consist of minimal Fortran source for the required declarations.
The order will match the order they first appeared in the source.

Some normalization will take place:
- extraneous spaces will be removed
- implicit types will be made explicit
- attributes will be written in a consistent order
- entity declarations will be combined into a single declaration
- function return types specified in a *prefix-spec* will be replaced by
  an entity declaration
- etc.

#### Symbols included

All public symbols from the module need to be included.

In addition, some private symbols are needed:
- private types that appear in the public API
- private components of non-private derived types
- private parameters used in non-private declarations (initial values, kind parameters)
- others?

It might be possible to anonymize private names if users don't want them exposed
in the `.mod` file. (Currently they are readable in PGI `.mod` files.)

#### USE association

A module that contains `USE` statements needs them represented in the
`.mod` file.
Each use-associated symbol will be written as a separate *use-only* statement,
possibly with renaming.

Alternatives:
- Emit a single `USE` for each module, listing all of the symbols that were
  use-associated in the *only-list*.
- Detect when all of the symbols from a module are imported (either by a *use-stmt*
  without an *only-list* or because all of the public symbols of the module
  have been listed in *only-list*s). In that case collapse them into a single *use-stmt*.
- Emit the *use-stmt*s that appeared in the original source.

## Reading and writing module files

### Options

The compiler will have command-line options to specify where to search
for module files and where to write them. By default it will be the current
directory for both.

For PGI, `-I` specifies directories to search for include files and module
files. `-module` specifics a directory to write module files in as well as to
search for them. gfortran is similar except it uses `-J` instead of `-module`.

The search order for module files is:
1. The `-module` directory (Note: for gfortran the `-J` directory is not searched).
2. The current directory
3. The `-I` directories in the order they appear on the command line

### Writing module files

When writing a module file, if the existing one matches what would be written,
the timestamp is not updated.

Module files will be written after semantics, i.e. after the compiler has
determined the module is valid Fortran.<br>
**NOTE:** PGI does create `.mod` files sometimes even when the module has a
compilation error.

Question: If the compiler can get far enough to determine it is compiling a module
but then encounters an error, should it delete the existing `.mod` file?
PGI does not, gfortran does.

### Reading module files

When the compiler finds a `.mod` file it needs to read, it firsts checks the first
line and verifies it is a valid module file. It can also verify checksums of
modules it depends on and report if they are out of date.

If the header is valid, the module file will be run through the parser and name
resolution to recreate the symbols from the module. Once the symbol table is
populated the parse tree can be discarded.

When processing `.mod` files we know they are valid Fortran with these properties:
1. The input (without the header) is already in the "cooked input" format.
2. No preprocessing is necessary.
3. No errors can occur.

## Error messages referring to modules

With this design, diagnostics can refer to names in modules and can emit a
normalized declaration of an entity but not point to its location in the
source.

If the header includes the source file it came from, that could be included in
a diagnostic but we still wouldn't have line numbers.

To provide line numbers and character positions or source lines as the user
wrote them we would have to save some amount of provenance information in the
module file as well.
