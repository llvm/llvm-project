====================================
Printf Behavior Under All Conditions
====================================

Introduction: 
=============
On the "defining undefined behavior" page, I said you should write down your
decisions regarding undefined behavior in your functions. This is that document
for my printf implementation.

Unless otherwise specified, the functionality described is aligned with the ISO
C standard and POSIX standard. If any behavior is not mentioned here, it should
be assumed to follow the behavior described in those standards.

The LLVM-libc codebase is under active development, and may change. This
document was last updated [August 18, 2023] by [michaelrj] and may
not be accurate after this point.

The behavior of LLVM-libc's printf is heavily influenced by compile-time flags.
Make sure to check what flags are defined before filing a bug report. It is also
not relevant to any other libc implementation of printf, which may or may not
share the same behavior.

This document assumes familiarity with the definition of the printf function and
is intended as a reference, not a replacement for the original standards.

--------------
General Flags:
--------------
These compile-time flags will change the behavior of LLVM-libc's printf when it
is compiled. Combinations of flags that are incompatible will be marked.

LIBC_COPT_STDIO_USE_SYSTEM_FILE
-------------------------------
When set, this flag changes fprintf and printf to use the FILE API from the
system's libc, instead of LLVM-libc's internal FILE API. This is set by default
when LLVM-libc is built in overlay mode.

LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
-----------------------------------
When set, this flag disables support for the POSIX "%n$" format, hereafter
referred to as "index mode"; conversions using the index mode format will be
treated as invalid. This reduces code size.

LIBC_COPT_PRINTF_INDEX_ARR_LEN
------------------------------
This flag takes a positive integer value, defaulting to 128. This flag
determines the number of entries the parser's type descriptor array has. This is
used in index mode to avoid re-parsing the format string to determine types when
an index lower than the previously specified one is requested. This has no
effect when index mode is disabled.

LIBC_COPT_PRINTF_DISABLE_WRITE_INT
----------------------------------
When set, this flag disables support for the C Standard "%n" conversion; any
"%n" conversion will be treated as invalid. This is set by default to improve
security.

LIBC_COPT_PRINTF_DISABLE_FLOAT
------------------------------
When set, this flag disables support for floating point numbers and all their
conversions (%a, %f, %e, %g); any floating point number conversion will be
treated as invalid. This reduces code size.

LIBC_COPT_PRINTF_NO_NULLPTR_CHECKS
----------------------------------
When set, this flag disables the nullptr checks in %n and %s.

LIBC_COPT_PRINTF_CONV_ATLAS
---------------------------
When set, this flag changes the include path for the "converter atlas" which is
a header that includes all the files containing the conversion functions. This
is not recommended to be set without careful consideration.

LIBC_COPT_PRINTF_HEX_LONG_DOUBLE
--------------------------------
When set, this flag replaces all decimal long double conversions (%Lf, %Le, %Lg)
with hexadecimal long double conversions (%La). This will improve performance
significantly, but may cause some tests to fail. This has no effect when float
conversions are disabled.

--------------------------------
Float Conversion Internal Flags:
--------------------------------
The following floating point conversion flags are provided for reference, but
are not recommended to be adjusted except by persons familiar with the Printf
Ryu Algorithm. Additionally they have no effect when float conversions are
disabled.

LIBC_COPT_FLOAT_TO_STR_USE_MEGA_LONG_DOUBLE_TABLE
-------------------------------------------------
When set, the float to string decimal conversion algorithm will use a larger
table to accelerate long double conversions. This larger table is around 5MB of 
size when compiled. This flag is enabled by default in the CMake.

LIBC_COPT_FLOAT_TO_STR_USE_DYADIC_FLOAT(_LD)
--------------------------------------------
When set, the float to string decimal conversion algorithm will use dyadic
floats instead of a table when performing floating point conversions. This
results in ~50 digits of accuracy in the result, then zeroes for the remaining
values. This may improve performance but may also cause some tests to fail. The
flag ending in _LD is the same, but only applies to long double decimal
conversions.

LIBC_COPT_FLOAT_TO_STR_USE_INT_CALC
-----------------------------------
When set, the float to string decimal conversion algorithm will use wide
integers instead of a table when performing floating point conversions. This
gives the same results as the table, but is very slow at the extreme ends of
the long double range. If no flags are set this is the default behavior for
long double conversions.

LIBC_COPT_FLOAT_TO_STR_NO_TABLE
-------------------------------
When set, the float to string decimal conversion algorithm will not use either
the mega table or the normal table for any conversions. Instead it will set
algorithmic constants to improve performance when using calculation algorithms.
If this flag is set without any calculation algorithm flag set, an error will
occur.

--------
Parsing:
--------

When printf encounters an invalid conversion specification, the entire
conversion specification will be passed literally to the output string.
As an example, printf("%Z") would display "%Z".

If an index mode conversion is requested for index "n" and there exists a number
in [1,n) that does not have a conversion specified in the format string, then
the conversion for index "n" is considered invalid.

If a non-index mode (also referred to as sequential mode) conversion is
specified after an index mode conversion, the next argument will be read but the
current index will not be incremented. From this point on, the arguments
selected by each conversion may or may not be correct. This is considered
dangerously undefined and may change without warning.

If a conversion specification is provided an invalid type modifier, that type
modifier will be ignored, and the default type for that conversion will be used.
In the case of the length modifier "L" and integer conversions, it will be
treated as if it was "ll" (lowercase LL). For this purpose the list of integer
conversions is d, i, u, o, x, X, n.

If a conversion specification ending in % has any options that consume arguments
(e.g. "%*.*%") those arguments will be consumed as normal, but their values will
be ignored.

If a conversion specification ends in a null byte ('\0') then it shall be
treated as an invalid conversion followed by a null byte.

If a number passed as a min width or precision value is out of range for an int,
then it will be treated as the largest or smallest value in the int range
(e.g. "%-999999999999.999999999999s" is the same as "%-2147483648.2147483647s").

----------
Conversion
----------
Any conversion specification that contains a flag or option that it does not
have defined behavior for will ignore that flag or option (e.g. %.5c is the same
as %c).

If a conversion specification ends in %, then it will be treated as if it is
"%%", ignoring all options.

If a null pointer is passed to a %s conversion specification and null pointer
checks are enabled, it will be treated as if the provided string is "null".

If a null pointer is passed to a %n conversion specification and null pointer
checks are enabled, the conversion will fail and printf will return a negative
value.

If a null pointer is passed to a %p conversion specification, the string
"(nullptr)" will be returned instead of an integer value.

The %p conversion will display any non-null pointer as if it was a uintptr value
passed to a "%#tx" conversion, with all other options remaining the same as the
original conversion.

The %p conversion will display a null pointer as if it was the string
"(nullptr)" passed to a "%s" conversion, with all other options remaining the
same as the original conversion.
