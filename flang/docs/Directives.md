<!--===- docs/Directives.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Compiler directives supported by Flang

A list of non-standard directives supported by Flang

* `!dir$ fixed` and `!dir$ free` select Fortran source forms.  Their effect
  persists to the end of the current source file.
* `!dir$ ignore_tkr (tkr) var-list` omits checks on type, kind, and/or rank.
