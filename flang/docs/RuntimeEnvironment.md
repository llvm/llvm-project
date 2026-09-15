<!--===- docs/RuntimeEnvironment.md

   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

-->

```{contents}
---
local:
---
```

# Environment variables of significance to Fortran execution

A few environment variables are queried by the Fortran runtime support
library.

The following environment variables can affect the behavior of
Fortran programs during execution.

## `DEFAULT_UTF8=1`

Set `DEFAULT_UTF8` to cause formatted external input to assume UTF-8
encoding on input and use UTF-8 encoding on formatted external output.

## `FORT_CONVERT`

Determines the data conversions applied to all unformatted I/O units that do
not have an explicit `CONVERT=` specifier in their `OPEN` statements.
`FORT_CONVERT` is an alias for the `FORT_CONVERT_UNIT` environment
variable without any `exception`s.

`FORT_CONVERT=mode`
* `mode: 'NATIVE' | 'LITTLE_ENDIAN' | 'BIG_ENDIAN' | 'SWAP';`
* `NATIVE`: no conversions (default)
* `LITTLE_ENDIAN`: assumes that input is little-endian; emit little-endian
output
* `BIG_ENDIAN`: assumes that input is big-endian; emit big-endian output
* `SWAP`: reverses the endianness (always convert)

## `FORT_CONVERT_UNIT`

Determines the data conversion applied to specific unformatted I/O units.
 
```
FORT_CONVERT_UNIT= mode | mode ';' exception | exception ;
mode: 'NATIVE' | 'LITTLE_ENDIAN' | 'BIG_ENDIAN' | 'SWAP';
exception: mode ':' unit_list | unit_list ;
unit_list: unit_spec | unit_list ',' unit_spec ;
unit_spec: integer | integer '-' integer ;
integer: [0-9]+ ;
```

The endianness of unformatted files is determined in the following order:
1. The host processor's native endianness.
2. The setting of the `-fconvert=<mode>` flang compiler command-line option.
3. The global setting from the `FORT_CONVERT=mode` or `FORT_CONVERT_UNIT=mode`
environment variables.  Contradictory mode settings between `FORT_CONVERT` and
`FORT_CONVERT_UNIT` result in `FORT_CONVERT_UNIT` taking priority.
4. The explicit setting of the `CONVERT=` specifier in the `OPEN` statement for
a particular unit.
5. The exception setting for an individual unit or range of units from the
`FORT_CONVERT_UNIT` environment variable.

### Examples

* If the `FORT_CONVERT`, `FORT_CONVERT_UNIT`, and `CONVERT=` specifier in the
  `OPEN` statement are all missing, no data conversion is performed for
  unformatted I/O; the host processor's native encoding is used.
* If the environment variable `FORT_CONVERT=BIG_ENDIAN` is set and no
  `CONVERT=` specifier is present in the `OPEN` statement, input is assumed to
  be big-endian, and output is emitted in big-endian format.
* On a little-endian host, if the environment variable `FORT_CONVERT=SWAP` is
  set and the `CONVERT=LITTLE_ENDIAN` specifier is present in the `OPEN`
  statement, the `OPEN` statement takes precedence: input is assumed to be
  little-endian, and output is emitted in little-endian format.
* If unit 10 is opened on a little-endian host with the environment variable
  `FORT_CONVERT=SWAP`, `FORT_CONVERT_UNIT=BIG_ENDIAN:10`, and an `OPEN`
  statement for unit 10 with the `CONVERT=LITTLE_ENDIAN` specifier, the
  `BIG_ENDIAN:10` exception from the `FORT_CONVERT_UNIT` environment variable
  takes precedence: input is assumed to be big-endian, and output is emitted
  in big-endian format.

### Notes

1. `<mode>` values specified with the runtime environment variables
   `FORT_CONVERT` or `FORT_CONVERT_UNIT` are case-insensitive.
2. `<mode>` values specified with the flang command-line option
   `-fconvert=<mode>` are case-sensitive and include:  
`mode: 'native' | 'little-endian' | 'big-endian' | 'swap';`
3. `unit_spec` supports ranges separated by a hyphen. Ranges must denote
positive unit numbers, and the starting unit (LHS) must be less than or equal
to the ending unit (RHS).
4. Unit numbers and ranges can be specified multiple times with different
`modes`, with the last (rightmost) `exception` taking priority. For example:
`FORT_CONVERT_UNIT="LITTLE_ENDIAN:10,11,15-20;BIG_ENDIAN:19"`  
The conversion for unit 19 will be `BIG_ENDIAN`.
5. If an `exception` is not prefixed with `mode:`, `mode` is assumed to be
`BIG_ENDIAN`.  For example:  
`FORT_CONVERT_UNIT="20-25;LITTLE_ENDIAN:26-27"`  
Regardless of the host processor's endianness, units 20 through 25 will be
treated as big-endian for both input and output, while units 26 and 27 will be
treated as little-endian for both input and output.


## `FORT_CHECK_POINTER_DEALLOCATION`

Fortran requires that a pointer that appears in a `DEALLOCATE` statement
must have been allocated in an `ALLOCATE` statement with the same declared
type.
The runtime support library validates this requirement by checking the
size of the allocated data, and will fail with an error message if
the deallocated pointer is not valid.
Set `FORT_CHECK_POINTER_DEALLOCATION=0` to disable this check.

## `FORT_FMT_RECL`

Set to an integer value to specify the record length for list-directed
and `NAMELIST` output.
The default is 72.

## `NO_STOP_MESSAGE`

Set `NO_STOP_MESSAGE=1` to disable the extra information about
IEEE floating-point exception flags that the Fortran language
standard requires for `STOP` and `ERROR STOP` statements.

## `FORT_TRUNCATE_STREAM`

Set `FORT_TRUNCATE_STREAM=1` to make output to a formatted unit
with `ACCESS="STREAM"` truncate the file when the unit has been
repositioned via `POS=` to an earlier point in the file.
This behavior is analogous to the implicit writing of an ENDFILE record
when output takes place to a sequential unit after
executing a `BACKSPACE` or `REWIND` statement.
Truncation of a stream-access unit is common to several other
compilers, but it is not mentioned in the standard.

## `FORT_NO_EMPTY_ALLOCATION`

Set `FORT_NO_EMPTY_ALLOCATION=1` to cause `ALLOCATE` statements
fail when the allocated size is empty.

## `FLANG_TRAMPOLINE_POOL_SIZE`

Set `FLANG_TRAMPOLINE_POOL_SIZE` to an integer value to control the maximum
number of runtime trampoline slots available when `-fsafe-trampoline` is
enabled. Each slot consists of a small executable code stub (size varies by
target; e.g. 32 bytes on x86-64 and AArch64) backed by a writable data entry.
The default is 1024 slots, which is sufficient for typical Fortran
programs. If more internal-procedure closures are alive simultaneously than
the pool can hold, the runtime terminates with a diagnostic message that
includes the current pool capacity.

Example: `export FLANG_TRAMPOLINE_POOL_SIZE=4096`
