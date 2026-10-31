```{title} clang-tidy - bugprone-macro-condition
```

# bugprone-macro-condition

Warns about inconsistent macro usage in preprocessor conditions.

Given the following code:

```c++
#define USE_FOO 0
// ...
#if defined(USE_FOO)
  // ...
#endif
// ...
#if USE_FOO
  // ...
#endif
```

`USE_FOO` is checked for definition in one condition and for value in
another. Was the intention to evaluate `USE_FOO` for a `true` expression,
or was the intention to merely check whether the macro was defined?

The check compares uses only when they refer to the same active definition
of an object-like macro with a nonempty replacement list and occur in the
same physical source file. It emits at most one warning for each macro
definition in each file. Merely defining a macro with a value and testing
its definition does not produce a warning by itself.

## Excluded scenarios

The check excludes the following scenarios:

- Uses in different physical source files are not compared.
- Uses resolving to different active macro definitions are not compared.
  Consequently, uses separated by ``#undef`` or a redefinition are not
  combined.
- Undefined macros, including macros undefined with command-line ``-U``,
  are ignored. Value tests of undefined macros are handled by `-Wundef`.
- Function-like macros and function-like invocations are ignored. This
  includes preprocessing operators such as `__has_builtin`,
  `__has_include`, and `__has_cpp_attribute`, as well as identifiers in
  their argument lists.
- Compiler-provided builtin macros are ignored.
- By default, macros whose names begin with `__` are ignored because these
  identifiers are reserved for use by implementations.
- Object-like macros with empty replacement lists are ignored.
- Negated definition tests are ignored. These include ``#ifndef``,
  ``#elifndef``, and ``!defined(FEATURE)``.
- A compound condition that tests the same macro for both definition and
  value is treated as one coherent test and ignored.
- A value test nested in a same-file positive definition guard is ignored.
  Qualifying guards are ``#ifdef``, ``#elifdef``, and ``#if`` or ``#elif``
  expressions consisting solely of positive definition tests joined by
  ``&&``. Disjunctions and ``#else`` branches do not establish a guard.
- A positive definition test is ignored when its controlled branch uses
  the macro's replacement value. Such a test can legitimately guard an
  expansion that would be invalid when the macro is undefined, while other
  conditions independently evaluate the value.
- A guard that supplies a default value for its macro is not considered a
  definition test. This applies to both ``#ifndef FEATURE`` and
  ``#if !defined(FEATURE)`` forms.
- A condition whose branch contains a top-level ``#error`` directive is
  treated as configuration validation and ignored. An ``#error`` inside a
  nested conditional is not unconditional and does not suppress a warning.
- Non-macro language tokens such as `true`, `false`, and C++ operator
  keywords are not considered macro references.

For example, a value test nested inside a positive definition guard does
not qualify for a warning:

```c++
#ifdef FEATURE
#if FEATURE >= 2
  // ...
#endif
#endif
```

A guard that supplies a default value also does not qualify:

```c++
#ifndef FEATURE
#define FEATURE 0
#endif

#if FEATURE
  // ...
#endif
```

No fixes are offered because the intended semantics are ambiguous.

To resolve a warning, decide which property of the macro is important:

- If the macro's value is important, keep the value in its definition and
  refactor definition tests to test the value, for example with
  ``#if USE_FOO`` or an explicit comparison.
- If the macro's presence or absence is important, make it a presence-only
  macro and refactor value tests to use ``defined(USE_FOO)`` or
  ``!defined(USE_FOO)`` consistently.

## Options

```{option} CheckDoubleUnderscoreMacros
If `true`, macros whose names begin with `__` are analyzed. Such
identifiers are reserved for use by implementations and are ignored by
default.
Default is `false`.
```
