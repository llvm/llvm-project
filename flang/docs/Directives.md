<!--===- docs/Directives.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Compiler directives supported by Flang

A list of non-standard directives supported by Flang

* `!dir$ fixed` and `!dir$ free` select Fortran source forms.  Their effect
  persists to the end of the current source file.
* `!dir$ ignore_tkr [[(TKRDMAC)] dummy-arg-name]...` in an interface definition
  disables some semantic checks at call sites for the actual arguments that
  correspond to some named dummy arguments (or all of them, by default).
  The directive allow actual arguments that would otherwise be diagnosed
  as incompatible in type (T), kind (K), rank (R), CUDA device (D), or
  managed (M) status.  The letter (A) is a shorthand for all of these,
  and is the default when no letters appear.  The letter (C) is a legacy
  no-op.  For example, if one wanted to call a "set all bytes to zero"
  utility that could be applied to arrays of any type or rank:
```
  interface
    subroutine clear(arr,bytes)
!dir$ ignore_tkr arr
      integer(1), intent(out) :: arr(bytes)
    end
  end interface
```
