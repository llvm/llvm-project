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

Determines data conversions applied to unformatted I/O.

* `NATIVE`: no conversions (default)
* `LITTLE_ENDIAN`: assume input is little-endian; emit little-endian output
* `BIG_ENDIAN`: assume input is big-endian; emit big-endian output
* `SWAP`: reverse endianness (always convert)

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
