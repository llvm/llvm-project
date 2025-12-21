<!--===- docs/Aliasing.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

# Bijective Internal Name Uniquing

```{contents}
---
local:
---
```

FIR has a flat namespace. No two objects may have the same name at the module
level. (These would be functions, globals, etc.) This necessitates some sort
of encoding scheme to unique symbols from the front-end into FIR.

Another requirement is to be able to reverse these unique names and recover
the associated symbol in the symbol table.

Fortran is case insensitive, which allows the compiler to convert the user's
identifiers to all lower case. Such a universal conversion implies that all
upper case letters are available for use in uniquing.

## Prefix `_Q`

All uniqued names have the prefix sequence `_Q` to indicate the name has been
uniqued. (Q is chosen because it is a [low frequency letter](http://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html)
in English.)

## Scope Building

Symbols are scoped by any module, submodule, procedure, and block that
contains that symbol. After the `_Q` sigil, names are constructed from
outermost to innermost scope as

   * Module name prefixed with `M`
   * Submodule name/s prefixed with `S`
   * Procedure name/s prefixed with `F`
   * Innermost block index prefixed with `B`

Given:
```
    submodule (mod:s1mod) s2mod
      ...
      subroutine sub
        ...
      contains
        function fun
```

The uniqued name of `fun` becomes:
```
    _QMmodSs1modSs2modFsubPfun
```

## Prefix tag summary

| Tag | Description
| ----| --------------------------------------------------------- |
| B   | Block ("name" is a compiler generated integer index)
| C   | Common block
| D   | Dispatch table (compiler internal)
| E   | variable Entity
| EC  | Constant Entity
| F   | procedure/Function (as a prefix)
| K   | Kind
| KN  | Negative Kind
| M   | Module
| N   | Namelist group
| P   | Procedure/function (as itself)
| Q   | uniQue mangled name tag
| S   | Submodule
| T   | derived Type
| Y   | tYpe descriptor (compiler internal)
| YI  | tYpe descriptor for an Intrinsic type (compiler internal)

## Common blocks

   * A common block name will be prefixed with `C`

Given:
```
   common /work/ i, j
```

The uniqued name of `work` becomes:
```
    _QCwork
```

Given:
```
   common i, j
```

The uniqued name in case of `blank common block` becomes:
```
    _QC
```

## Module scope global data

   * A global data entity is prefixed with `E`
   * A global entity that is constant (parameter) will be prefixed with `EC`

Given:
```
    module mod
      integer :: intvar
      real, parameter :: pi = 3.14
    end module
```

The uniqued name of `intvar` becomes:
```
    _QMmodEintvar
```

The uniqued name of `pi` becomes:
```
    _QMmodECpi
```

## Procedures

   * A procedure/subprogram as itself is prefixed with `P`
   * A procedure/subprogram as an ancestor name is prefixed with `F`

Procedures are the only names that are themselves uniqued, as well as
appearing as a prefix component of other uniqued names.

Given:
```
    subroutine sub
      real, save :: x(1000)
      ...
```
The uniqued name of `sub` becomes:
```
    _QPsub
```
The uniqued name of `x` becomes:
```
    _QFsubEx
```

## Blocks

   * A block is prefixed with `B`; the block "name" is a compiler generated
     index

Each block has a per-procedure preorder index. The prefix for the immediately
containing block construct is unique within the procedure.

Given:
```
    subroutine sub
    block
      block
        real, save :: x(1000)
        ...
      end block
      ...
    end block
```
The uniqued name of `x` becomes:
```
    _QFsubB2Ex
```

## Namelist groups

   * A namelist group is prefixed with `N`

Given:
```
    subroutine sub
      real, save :: x(1000)
      namelist /temps/ x
      ...
```
The uniqued name of `temps` becomes:
```
    _QFsubNtemps
```

## Derived types

   * A derived type is prefixed with `T`
   * If a derived type has KIND parameters, they are listed in a consistent
     canonical order where each takes the form `Ki` and where _i_ is the
     compile-time constant value. (All type parameters are integer.)  If _i_
     is a negative value, the prefix `KN` will be used and _i_ will reflect
     the magnitude of the value.

Given:
```
    module mymodule
      type mytype
        integer :: member
      end type
      ...
```
The uniqued name of `mytype` becomes:
```
    _QMmymoduleTmytype
```

Given:
```
    type yourtype(k1,k2)
      integer, kind :: k1, k2
      real :: mem1
      complex :: mem2
    end type
```

The uniqued name of `yourtype` where `k1=4` and `k2=-6` (at compile-time):
```
    _QTyourtypeK4KN6
```

   * A derived type dispatch table is prefixed with `D`. The dispatch table
     for `type t` would be `_QDTt`
   * A type descriptor instance is prefixed with `C`. Intrinsic types can
     be encoded with their names and kinds. The type descriptor for the
     type `yourtype` above would be `_QCTyourtypeK4KN6`. The type
     descriptor for `REAL(4)` would be `_QCrealK4`.

## Compiler internal names

Compiler generated names do not have to be mapped back to Fortran. This
includes names prefixed with `_QQ`, tag `D` for a type bound procedure
dispatch table, and tags `Y` and `YI` for runtime type descriptors.
Combinations of internal names are separated with the `X` tag.

Given:
```
    _QQcl, 9a37c0
```

The uniqued name of `_QQcl` and `9a37c0`:
```
    _QQclX9a37c0
```
