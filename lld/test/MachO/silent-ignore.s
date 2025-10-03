# REQUIRES: x86

## Check that we correctly parse these flags, even though they are
## unimplemented. ## We may still emit warnings or errors for some of the
## unimplemented ones (but those ## errors are squelched because of the
## `--version` flag.)
# RUN: %lld --version \
# RUN:   -dynamic \
# RUN:   -lto_library /lib/foo \
# RUN:   -macosx_version_min 0 \
# RUN:   -no_dtrace_dof \
# RUN:   -dependency_info /path/to/dependency_info.dat \
# RUN:   -lto_library ../lib/libLTO.dylib \
# RUN:   -mllvm -time-passes \
# RUN:   -objc_abi_version 2 \
# RUN:   -ios_simulator_version_min 9.0.0 \
# RUN:   -sdk_version 13.2
# RUN: not %lld -v --not-an-ignored-argument 2>&1 | FileCheck %s
# CHECK: error: unknown argument '--not-an-ignored-argument'

## Check that we don't emit any warnings nor errors for these unimplemented flags.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld %t.o -o /dev/null -objc_abi_version 2 -debug_variant -no_warn_duplicate_libraries

.globl _main
_main:
  ret
