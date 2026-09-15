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

## `FLANG_RT_COPYOUT_MODIFIED_ONLY`

The system environment variable `FLANG_RT_COPYOUT_MODIFIED_ONLY` selects how
the runtime performs copy-out.

When the compiler passes a copy of an actual argument to a procedure
(copy-in/copy-out), the runtime skips the copy-out entirely when the
temporary copy is still bitwise-identical to the original, and performs
the normal whole-object copy-out otherwise. This avoids stores to the
original argument when the callee never modified the data -- in
particular, stores into read-only storage backing a non-definable actual
argument.
Set the system environment variable `FLANG_RT_COPYOUT_MODIFIED_ONLY=0` to
restore the unconditional copy-out.

## `FLANG_RT_COPYOUT_READONLY_MODE`

An optional compatibility mode (host only; default `0` = off). When enabled,
the runtime consults the process memory map and skips a copy-out whose
destination lies in read-only memory: such a store could only rewrite
identical bytes or crash, so skipping converts the crash into a no-op for
programs that (invalidly) modified a temporary whose original is not
definable.

* `1`: trust a one-time lazy snapshot of the memory map (restricted to
  file-backed private read-only mappings); no system calls on the copy-out
  path. A mapping whose protection changes after the snapshot is not seen:
  a region that became read-only is simply not recognized (the regular
  copy-out runs, as without this feature), and a formerly read-only region
  that became writable is still skipped (the copy-out is lost). Both are
  accepted, documented behaviors of this mode.
* `2`: additionally re-confirm each snapshot hit against the current memory
  map before skipping (system calls on hits only).

Set `FLANG_RT_COPYOUT_READONLY_DIAG=1` to report the first few skipped
copy-outs on standard error.

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
