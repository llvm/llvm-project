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
program units in other source files that depend on that module.
Program units in the same source file as the module do not read
module files, as this compiler parses entire source files and processes
the program units it contains in dependency order.

## Name

Module files are named according to the module's name, suffixed with `.mod`.
This is consistent with other compilers and expected by makefiles and
other build systems.

The disadvantage of using the same name as other compilers is that it is not
clear which compiler created a `.mod` file and files from multiple compilers
cannot be in the same directory. This can be solved by adding something
between the module name and extension, e.g. `<modulename>-f18.mod`.  When
this is needed, Flang accepts the option `-module-suffix` to alter the suffix.

## Format

Module files are Fortran free form source code.
(One can, in principle, copy `foo.mod` into `tmp.f90`, recompile it,
and obtain a matching `foo.mod` file.)
They include the declarations of all visible locally defined entities along
with the private entities on which thef depend.
Entity declarations that span multiple statements are collapsed into
a single *type-declaration-statement*.
Executable statements are omitted.

### Header

Module files begin with a UTF-8 byte order mark and a few lines of
Fortran comments.
(Pro tip: use `dd if=foo.mod bs=1 skip=3 2>/dev/null` to skip the byte order
mark and dump the rest of the module.)
The first comment begins `!mod$` and contains a version number
and hash code.
Further `!need$` comments contain the names and hash codes of other modules
on which this module depends, and whether those modules are intrinsic
or not to Fortran.

The header comments do not contain timestamps or original source file paths.

### Body

The body comprises  minimal Fortran source for the required declarations.
The order generally matches the order they appeared in the original
source code for the module.
All types are explicit, and all non-character literal constants are
marked with explicit kind values.

#### Symbols included

All public symbols from the module are included.

In addition, some private symbols are needed:
- private types that appear in the public API
- private components of non-private derived types
- private parameters used in non-private declarations (initial values, kind parameters)

#### USE association

A module that contains `USE` statements needs them represented in the
`.mod` file.
Each use-associated symbol will be written as a separate *use-only* statement,
possibly with renaming.

## Reading and writing module files

### Options

The compiler has command-line options to specify where to search
for module files and where to write them. By default it will be the current
directory for both.

`-I` specifies directories to search for include files and module
files. `-J` specifies a directory into which module files are written
as well as to search for them.

### Writing module files

When writing a module file, if the existing one matches what would be written,
the timestamp is not updated.

Module files are written only after semantic analysis completes without
a fatal error message.

### Reading module files

When the compiler finds a `.mod` file it needs to read, it firsts checks the first
line and verifies it is a valid module file.
The header checksum must match the file's contents.
(Pro tip: if a developer needs to hack the contents of a module file, they can
recompile it afterwards as Fortran source to regenerate it with its new hash.)

The known hashes of dependent modules are used to disambiguate modules whose
names match module files in multiple search directories, as well as to
detect dependent modules whose recompilation has rendered a module file
obsolete.

When processing `.mod` files we know they are valid Fortran with these properties:
1. The input (without the header) is already in the "cooked input" format.
2. No preprocessing is necessary.
3. No errors can occur.

## Error messages referring to modules

With this design, diagnostics can refer to names in modules and can emit a
normalized declaration of an entity.

## Hermetic modules files

Top-level module files for libraries can be build with `-fhermetic-module-files`.
This option causes these module files to contain copies of all of the non-intrinsic
modules on which they depend, so that non-top-level local modules and the
modules of dependent libraries need not also be packaged with the library.
When the compiler reads a hermetic module file, the copies of the dependent
modules are read into their own scope, and will not conflict with other modules
of the same name that client code might `USE`.
