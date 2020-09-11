<!--===- docs/Preprocessing.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Fortran Preprocessing

```eval_rst
.. contents::
   :local:
```

## Behavior common to (nearly) all compilers:

* Macro and argument names are sensitive to case.
* Fixed form right margin clipping after column 72 (or 132)
  has precedence over macro name recognition, and also over
  recognition of function-like parentheses and arguments.
* Fixed form right margin clipping does not apply to directive lines.
* Macro names are not recognized as such when spaces are inserted
  into their invocations in fixed form.
  This includes spaces at the ends of lines that have been clipped
  at column 72 (or whatever).
* Text is rescanned after expansion of macros and arguments.
* Macros are not expanded within quoted character literals or
  quoted FORMAT edit descriptors.
* Macro expansion occurs before any effective token pasting via fixed form
  space removal.
* C-like line continuations with backslash-newline are allowed in
  directives, including the definitions of macro bodies.
* `/* Old style C comments */` are ignored in directives and
  removed from the bodies of macro definitions.
* `// New style C comments` are not removed, since Fortran has OPERATOR(//).
* C-like line continuations with backslash-newline can appear in
  old-style C comments in directives.
* After `#define FALSE TRUE`, `.FALSE.` is replaced by `.TRUE.`;
  i.e., tokenization does not hide the names of operators or logical constants.
* `#define KWM c` allows the use of `KWM` in column 1 as a fixed form comment
  line indicator.
* A `#define` directive intermixed with continuation lines can't
  define a macro that's invoked earlier in the same continued statement.

## Behavior that is not consistent over all extant compilers but which probably should be uncontroversial:

* Invoked macro names can straddle a Fortran line continuation.
* ... unless implicit fixed form card padding intervenes; i.e.,
  in fixed form, a continued macro name has to be split at column
  72 (or 132).
* Comment lines may appear with continuations in a split macro names.
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the name and the left parenthesis, and comment and
  directive lines can be there too.
* Function-like macro invocations can straddle a Fortran fixed form line
  continuation between the parentheses, and comment lines can be there too.
* Macros are not expanded within Hollerith constants or Hollerith
  FORMAT edit descriptors.
* Token pasting with `##` works in function-like macros.
* Argument stringization with `#` works in function-like macros.
* Directives can be capitalized (e.g., `#DEFINE`) in fixed form.
* Fixed form clipping after column 72 or 132 is done before macro expansion,
  not after.
* C-like line continuation with backslash-newline can appear in the name of
  a keyword-like macro definition.
* If `#` is in column 6 in fixed form, it's a continuation marker, not a
  directive indicator.
* `#define KWM !` allows KWM to signal a comment.

## Judgement calls, where precedents are unclear:

* Expressions in `#if` and `#elif` should support both Fortran and C
  operators; e.g., `#if 2 .LT. 3` should work.
* If a function-like macro does not close its parentheses, line
  continuation should be assumed.
* ... However, the leading parenthesis has to be on the same line as
  the name of the function-like macro, or on a continuation line thereof.
* If macros expand to text containing `&`, it doesn't work as a free form
  line continuation marker.
* `#define c 1` does not allow a `c` in column 1 to be used as a label
  in fixed form, rather than as a comment line indicator.
* IBM claims to be ISO C compliant and therefore recognizes trigraph sequences.
* Fortran comments in macro actual arguments should be respected, on
  the principle that a macro call should work like a function reference.
* If a `#define` or `#undef` directive appears among continuation
  lines, it may or may not affect text in the continued statement that
  appeared before the directive.

## Behavior that few compilers properly support (or none), but should:

* A macro invocation can straddle free form continuation lines in all of their
  forms, with continuation allowed in the name, before the arguments, and
  within the arguments.
* Directives can be capitalized in free form, too.
* `__VA_ARGS__` and `__VA_OPT__` work in variadic function-like macros.

## In short, a Fortran preprocessor should work as if:

1. Fixed form lines are padded up to column 72 (or 132) and clipped thereafter.
2. Fortran comments are removed.
3. C-style line continuations are processed in preprocessing directives.
4. C old-style comments are removed from directives.
5. Fortran line continuations are processed (outside preprocessing directives).
   Line continuation rules depend on source form.
   Comment lines that are enabled compiler directives have their line
   continuations processed.
   Conditional compilation preprocessing directives (e.g., `#if`) may be
   appear among continuation lines, and have their usual effects upon them.
6. Other preprocessing directives are processed and macros expanded.
   Along the way, Fortran `INCLUDE` lines and preprocessor `#include` directives
   are expanded, and all these steps applied recursively to the introduced text.
7. Any Fortran comments created by macro replacement are removed.

Steps 5 and 6 are interleaved with respect to the preprocessing state.
Conditional compilation preprocessing directives always reflect only the macro
definition state produced by the active `#define` and `#undef` preprocessing directives
that precede them.

If the source form is changed by means of a compiler directive (i.e.,
`!DIR$ FIXED` or `FREE`) in an included source file, its effects cease
at the end of that file.

Last, if the preprocessor is not integrated into the Fortran compiler,
new Fortran continuation line markers should be introduced into the final
text.

OpenMP-style directives that look like comments are not addressed by
this scheme but are obvious extensions.

## Appendix
`N` in the table below means "not supported"; this doesn't
mean a bug, it just means that a particular behavior was
not observed.
`E` signifies "error reported".

The abbreviation `KWM` stands for "keyword macro" and `FLM` means
"function-like macro".

The first block of tests (`pp0*.F`) are all fixed-form source files;
the second block (`pp1*.F90`) are free-form source files.

```
f18
| pgfortran
| | ifort
| | | gfortran
| | | | xlf
| | | | | nagfor
| | | | | |
. . . . . .   pp001.F  keyword macros
. . . . . .   pp002.F  #undef
. . . . . .   pp003.F  function-like macros
. . . . . .   pp004.F  KWMs case-sensitive
. N . N N .   pp005.F  KWM split across continuation, implicit padding
. N . N N .   pp006.F  ditto, but with intervening *comment line
N N N N N N   pp007.F  KWM split across continuation, clipped after column 72
. . . . . .   pp008.F  KWM with spaces in name at invocation NOT replaced
. N . N N .   pp009.F  FLM call split across continuation, implicit padding
. N . N N .   pp010.F  ditto, but with intervening *comment line
N N N N N N   pp011.F  FLM call name split across continuation, clipped
. N . N N .   pp012.F  FLM call name split across continuation
. E . N N .   pp013.F  FLM call split between name and (
. N . N N .   pp014.F  FLM call split between name and (, with intervening *comment
. E . N N .   pp015.F  FLM call split between name and (, clipped
. E . N N .   pp016.F  FLM call split between name and ( and in argument
. . . . . .   pp017.F  KLM rescan
. . . . . .   pp018.F  KLM rescan with #undef (so rescan is after expansion)
. . . . . .   pp019.F  FLM rescan
. . . . . .   pp020.F  FLM expansion of argument
. . . . . .   pp021.F  KWM NOT expanded in 'literal'
. . . . . .   pp022.F  KWM NOT expanded in "literal"
. . E E . E   pp023.F  KWM NOT expanded in 9HHOLLERITH literal
. . . E . .   pp024.F  KWM NOT expanded in Hollerith in FORMAT
. . . . . .   pp025.F  KWM expansion is before token pasting due to fixed-form space removal
. . . E . E   pp026.F  ## token pasting works in FLM
E . . E E .   pp027.F  #DEFINE works in fixed form
. N . N N .   pp028.F  fixed-form clipping done before KWM expansion on source line
. . . . . .   pp029.F  \ newline allowed in #define
. . . . . .   pp030.F  /* C comment */ erased from #define
E E E E E E   pp031.F   // C++ comment NOT erased from #define
. . . . . .   pp032.F  /* C comment */ \ newline erased from #define
. . . . . .   pp033.F  /* C comment \ newline */ erased from #define
. . . . . N   pp034.F  \ newline allowed in name on KWM definition
. E . E E .   pp035.F  #if 2 .LT. 3 works
. . . . . .   pp036.F  #define FALSE TRUE ...  .FALSE. -> .TRUE.
N N N N N N   pp037.F  fixed-form clipping NOT applied to #define
. . E . E E   pp038.F  FLM call with closing ')' on next line (not a continuation)
E . E . E E   pp039.F  FLM call with '(' on next line (not a continuation)
. . . . . .   pp040.F  #define KWM c, then KWM works as comment line initiator
E . E . . E   pp041.F  use KWM expansion as continuation indicators
N N N . . N   pp042.F  #define c 1, then use c as label in fixed-form
. . . . N .   pp043.F  #define with # in column 6 is a continuation line in fixed-form
E . . . . .   pp044.F  #define directive amid continuations
. . . . . .   pp101.F90  keyword macros
. . . . . .   pp102.F90  #undef
. . . . . .   pp103.F90  function-like macros
. . . . . .   pp104.F90  KWMs case-sensitive
. N N N N N   pp105.F90  KWM call name split across continuation, with leading &
. N N N N N   pp106.F90  ditto, with & ! comment
N N E E N .   pp107.F90  KWM call name split across continuation, no leading &, with & ! comment
N N E E N .   pp108.F90  ditto, but without & ! comment
. N N N N N   pp109.F90  FLM call name split with leading &
. N N N N N   pp110.F90  ditto, with & ! comment
N N E E N .   pp111.F90  FLM call name split across continuation, no leading &, with & ! comment
N N E E N .   pp112.F90  ditto, but without & ! comment
. N N N N E   pp113.F90  FLM call split across continuation between name and (, leading &
. N N N N E   pp114.F90  ditto, with & ! comment, leading &
N N N N N .   pp115.F90  ditto, with & ! comment, no leading &
N N N N N .   pp116.F90  FLM call split between name and (, no leading &
. . . . . .   pp117.F90  KWM rescan
. . . . . .   pp118.F90  KWM rescan with #undef, proving rescan after expansion
. . . . . .   pp119.F90  FLM rescan
. . . . . .   pp120.F90  FLM expansion of argument
. . . . . .   pp121.F90  KWM NOT expanded in 'literal'
. . . . . .   pp122.F90  KWM NOT expanded in "literal"
. . E E . E   pp123.F90  KWM NOT expanded in Hollerith literal
. . E E . E   pp124.F90  KWM NOT expanded in Hollerith in FORMAT
E . . E E .   pp125.F90  #DEFINE works in free form
. . . . . .   pp126.F90  \ newline works in #define
N . E . E E   pp127.F90  FLM call with closing ')' on next line (not a continuation)
E . E . E E   pp128.F90  FLM call with '(' on next line (not a continuation)
. . N . . N   pp129.F90  #define KWM !, then KWM works as comment line initiator
E . E . . E   pp130.F90  #define KWM &, use for continuation w/o pasting (ifort and nag seem to continue #define)
```
