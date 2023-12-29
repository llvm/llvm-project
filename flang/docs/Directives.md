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

## ARM Streaming SVE directives

These directives are added to support ARM specific instructions. All of
these attributes apply to a specific subroutine or function. These directives
are identical to the attributes provided in C and C++ for the same purpose.
See https://arm-software.github.io/acle/main/acle.html#controlling-the-use-of-streaming-mode for more in depth details. (For the following, function is used
to mean both subroutine and function).

### Directives relating to ARM Streaming mode

* `!dir$ arm_streaming` - The function is intended to be used in streaming
  mode.
* `!dir$ arm_streaming_compatible` - The function can work both in streaming
  mode and non-streaming mode.
* `!dir$ arm_streaming` - The function will enter streaming mode, and return to
  non-streaming mode when reaturning.

### Directives relating to ZA

* `!dir$ arm_shared_za` - A function that uses ZA for input or output.
* `!dir$ arm_new_za` - A function that has ZA state created and destroyed within
  the function.
* `!dir$ arm_preserves_za` - Optimisation hint for the compiler that the
  function either doesn't alter, or saves and restores the ZA state.
