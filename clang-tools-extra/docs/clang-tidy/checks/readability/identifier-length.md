```{title} clang-tidy - readability-identifier-length
```

# readability-identifier-length

This check finds variables and function parameters whose length are too short.
The desired name length is configurable. Special short names which should be
ignored can be specified. Local variables with short names can also be ignored
if they are short-lived.

## Options

The following options are described below:

- [`MinimumVariableNameLength`](#readability-identifier-length-minimum-variable-name-length),
  [`IgnoredVariableNames`](#readability-identifier-length-ignored-variable-names)
- [`MinimumBindingNameLength`](#readability-identifier-length-minimum-binding-name-length),
  [`IgnoredBindingNames`](#readability-identifier-length-ignored-binding-names)
- [`MinimumParameterNameLength`](#readability-identifier-length-minimum-parameter-name-length),
  [`IgnoredParameterNames`](#readability-identifier-length-ignored-parameter-names)
- [`MinimumLoopCounterNameLength`](#readability-identifier-length-minimum-loop-counter-name-length),
  [`IgnoredLoopCounterNames`](#readability-identifier-length-ignored-loop-counter-names)
- [`MinimumExceptionNameLength`](#readability-identifier-length-minimum-exception-name-length),
  [`IgnoredExceptionVariableNames`](#readability-identifier-length-ignored-exception-variable-names)
- [`LineCountThreshold`](#readability-identifier-length-line-count-threshold)

(readability-identifier-length-minimum-variable-name-length)=

````{option} MinimumVariableNameLength
All variables (other than loop counter, exception names and function
parameters) are expected to have at least a length of
[`MinimumVariableNameLength`](#readability-identifier-length-minimum-variable-name-length).
Setting it to `0` or `1` disables the check entirely. Default is `3`.

```c++
int i = 42;    // warns that 'i' is too short
```
````

(readability-identifier-length-ignored-variable-names)=

```{option} IgnoredVariableNames
Specifies a regular expression for variable names that are
to be ignored. Default is empty string, so no names are ignored.
```

(readability-identifier-length-minimum-binding-name-length)=

````{option} MinimumBindingNameLength
All variables introduced by structured bindings are expected to have at
least a length of
[`MinimumBindingNameLength`](#readability-identifier-length-minimum-binding-name-length).
Setting it to `0` or `1` disables the check entirely. Default is `2`.

```c++
auto [a] = get_result();    // warns that 'a' is too short
```
````

(readability-identifier-length-ignored-binding-names)=

```{option} IgnoredBindingNames
Specifies a regular expression for variable names introduced by structured
bindings that are to be ignored. The `^[_]$` value allows the `_` idiom to
specify that the value is discarded on purpose. Default is `^[_]$`.
```

(readability-identifier-length-minimum-parameter-name-length)=

````{option} MinimumParameterNameLength
All function parameter names are expected to have a length of at least
[`MinimumParameterNameLength`](#readability-identifier-length-minimum-parameter-name-length).
Setting it to `0` or `1` disables the check entirely. Default is `3`.

```c++
int doubler(int x)   // warns that x is too short
{
   return 2 * x;
}
```
````

(readability-identifier-length-ignored-parameter-names)=

```{option} IgnoredParameterNames
Specifies a regular expression for parameters that are to be ignored.
Default is `^[n]$` for historical reasons.
```

(readability-identifier-length-minimum-loop-counter-name-length)=

````{option} MinimumLoopCounterNameLength
Loop counter variables are expected to have a length of at least
[`MinimumLoopCounterNameLength`](#readability-identifier-length-minimum-loop-counter-name-length)
characters. Setting it to `0` or `1` disables the check entirely. Default is
`2`.

```c++
// This warns that 'q' is too short.
for (int q = 0; q < size; ++ q) {
   // ...
}
```
````

(readability-identifier-length-ignored-loop-counter-names)=

````{option} IgnoredLoopCounterNames
Specifies a regular expression for counter names that are to be ignored.
Default is `^[ijk_]$`; the first three symbols are included for historical
reasons and the last one since it is frequently used as a "don't care"
value, specifically in tools such as Google Benchmark.

```c++
// This does not warn by default, for historical reasons.
for (int i = 0; i < size; ++ i) {
    // ...
}
```
````

(readability-identifier-length-minimum-exception-name-length)=

````{option} MinimumExceptionNameLength
Exception clause variables are expected to have a length of at least
[`MinimumExceptionNameLength`](#readability-identifier-length-minimum-exception-name-length).
Setting it to `0` or `1` disables the check entirely. Default is `2`.

```c++
try {
    // ...
}
// This warns that 'e' is too short.
catch (const std::exception& x) {
    // ...
}
```
````

(readability-identifier-length-ignored-exception-variable-names)=

````{option} IgnoredExceptionVariableNames
Specifies a regular expression for exception variable names that are to
be ignored. Default is `^[e]$` mainly for historical reasons.

```c++
try {
    // ...
}
// This does not warn by default, for historical reasons.
catch (const std::exception& e) {
    // ...
}
```
````

(readability-identifier-length-line-count-threshold)=

````{option} LineCountThreshold
Defines the minimum number of lines required between declaration and last
use for a diagnostic to be issued. This option only affects the behavior
regarding local variables: a warning is always issued when a global variable
has a short name, because globals can potentially be used across multiple
files. Default is `0`, which corresponds to all variables being flagged.

```c++
// In this example, a warning will be issued if LineCountThreshold < N
int a = 0;      // First line (declaration line)
a = 1;          // Second line
                // ...
last_use_of(a); // N-th line
```
````
