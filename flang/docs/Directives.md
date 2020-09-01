Compiler directives supported by F18
====================================

* `!dir$ fixed` and `!dir$ free` select Fortran source forms.  Their effect
  persists to the end of the current source file.
* `!dir$ ignore_tkr (tkr) var-list` omits checks on type, kind, and/or rank.
