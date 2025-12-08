# Analysis of GitHub Issue 170591

**Issue Title:** `[flang] _FortranAExit does not have [[noreturn]] semantic`
**Issue URL:** https://github.com/llvm/llvm-project/issues/170591

## Summary of the Issue:
The user reported that the `flang` compiler does not correctly optimize code following a call to the intrinsic `exit()` (which translates to `_FortranAExit`). Specifically, at optimization level `-O3`, code that should be unreachable after `exit()` is still generated in the assembly output. This indicates that `flang` does not recognize `_FortranAExit` as a `noreturn` function.

**Expected Behavior:** Code after `call exit(status)` should be optimized out because the program execution terminates at `exit()`.
**Actual Behavior:** The `flang` compiler, even with `-O3`, generates assembly for statements that follow `call exit(status)`.

## Reproducer(s):
The issue body provided a Fortran function `KOHb_exit`. For local reproduction and detailed analysis, this function was isolated and compiled.

**`repro-main.f90`:**
```fortran
integer function KOHb_exit(status)
  integer, intent(in) :: status
  if (status > 0) call exit(status) ! actually, _FortranAExit
  KOHb_exit = 0
  do i = 1, status
    print '(A,I0)', "KOHb #", i
  end do
end function KOHb_exit
```

## Runner Script:
A `run-main.sh` script was created to compile `repro-main.f90` with `flang -O3 -S` to generate assembly code.

**`run-main.sh`:**
```bash
#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPRO_DIR="$SCRIPT_DIR"

FLANG_COMPILER="/home/eepshteyn/compilers/flang-upstream/bin/flang"

echo "Compiling repro-main.f90 with -O3..."
"$FLANG_COMPILER" -O3 -S "$REPRO_DIR/repro-main.f90" -o "$REPRO_DIR/repro-main.s"

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Assembly generated in repro-main.s"
```

## Compiler Output Mentioned in the Issue:
The issue mentions that at `-O3`, `flang` generates `print` statements in the assembly after `exit`. This was demonstrated via a Godbolt link showing the assembly output.

## Check if the issue is still present in local compiler:
The `run-main.sh` script was executed, generating `repro-main.s`. Analysis of `repro-main.s` revealed the following:
1.  The `_FortranAExit` call is present, correctly branched to based on the `status` argument.
2.  Immediately following the `callq _FortranAExit@PLT` instruction, the assembly code still contains instructions related to the `do` loop and `print` statement. These include calls to Fortran I/O routines such as `_FortranAioBeginExternalFormattedOutput`, `_FortranAioOutputAscii`, `_FortranAioOutputInteger32`, and `_FortranAioEndIoStatement`. The string literal "KOHb #" is also referenced.

This confirms that the compiler *does not* optimize away the unreachable code after `_FortranAExit`.

## Conclusion:
The issue reported in GitHub issue 170591 is still present in the local `flang` compiler (version 22.0.0, commit e74b425ddcac22ccc4d0bd5d65f95ffc2682b62f). The compiler fails to apply the `[[noreturn]]` semantic to `_FortranAExit`, leading to the generation of unreachable code in optimized assembly.
