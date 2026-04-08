! Test for errors and warnings generated when parsing driver options. You can
! use this file for relatively small tests and to avoid creating new test files.

! RUN: %flang -### -S -O4 -ffp-contract=on %s 2>&1 | FileCheck %s

! CHECK: warning: the argument 'on' is not supported for option 'ffp-contract='. Mapping to 'ffp-contract=off'
! CHECK: warning: -O4 is equivalent to -O3
! CHECK-LABEL: "-fc1"
! CHECK: -ffp-contract=off
! CHECK: -O3

! RUN: %flang -### -S -fprofile-generate %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-LLVM %s
! CHECK-PROFILE-GENERATE-LLVM: "-fprofile-generate"
! RUN: %flang -### -S -fprofile-use=%S %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-DIR %s
! CHECK-PROFILE-USE-DIR: "-fprofile-use={{.*}}"
!
! ------------------------------------------------------------------------------
! RUN: %flang -### -fbuiltin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-BUILTIN
! WARN-BUILTIN: warning: '-fbuiltin' is not valid for Fortran
!
! RUN: %flang -### -fno-builtin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-NO-BUILTIN
! WARN-NO-BUILTIN: warning: '-fno-builtin' is not valid for Fortran
!
! RUN: %flang -### -fbuiltin -fno-builtin %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix=WARN-BUILTIN-MULTIPLE
! WARN-BUILTIN-MULTIPLE: warning: '-fbuiltin' is not valid for Fortran
! WARN-BUILTIN-MULTIPLE: warning: '-fno-builtin' is not valid for Fortran
!
! ------------------------------------------------------------------------------
! When emitting an error with a suggestion, ensure that the diagnostic message
! uses '-Xflang' instead of '-Xclang'. This is typically emitted when an option
! that is available for `flang -fc1` is passed to `flang`. We use -complex-range
! since it is only available for fc1. If this option is ever exposed to `flang`,
! a different option will have to be used in the test below.
!
! RUN: not %flang -### -complex-range=full %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNKNOWN-SUGGEST
!
! UNKNOWN-SUGGEST: error: unknown argument '-complex-range=full';
! UNKNOWN-SUGGEST-SAME: did you mean '-Xflang -complex-range=full'
!
! RUN: not %flang -### -not-an-option %s 2>&1 \
! RUN:     | FileCheck %s -check-prefix UNKNOWN-NO-SUGGEST
!
! UNKNOWN-NO-SUGGEST: error: unknown argument: '-not-an-option'{{$}}
!
! ------------------------------------------------------------------------------
! The options in the command below are gfortran-specific optimization flags that
! are accepted by flang's driver but ignored. Check that the correct warning
! message is displayed when these are used.
! RUN: %flang -### %s                                                       \
! RUN:     -finline-limit=1000                                              \
! RUN:     -finline-limit                                                   \
! RUN:     -fexpensive-optimizations                                        \
! RUN:     -fno-expensive-optimizations                                     \
! RUN:     -fno-defer-pop                                                   \
! RUN:     -fkeep-inline-functions                                          \
! RUN:     -fno-keep-inline-functions                                       \
! RUN:     -freorder-blocks                                                 \
! RUN:     -ffloat-store                                                    \
! RUN:     -fgcse                                                           \
! RUN:     -fivopts                                                         \
! RUN:     -fprefetch-loop-arrays                                           \
! RUN:     -fprofile-correction                                             \
! RUN:     -fprofile-values                                                 \
! RUN:     -fschedule-insns                                                 \
! RUN:     -fsignaling-nans                                                 \
! RUN:     -fstrength-reduce                                                \
! RUN:     -ftracer                                                         \
! RUN:     -funroll-all-loops                                               \
! RUN:     -funswitch-loops                                                 \
! RUN:     -falign-labels                                                   \
! RUN:     -falign-labels=100                                               \
! RUN:     -falign-jumps                                                    \
! RUN:     -falign-jumps=100                                                \
! RUN:     -fbranch-count-reg                                               \
! RUN:     -fcaller-saves                                                   \
! RUN:     -fno-default-inline                                              \
! RUN:     -fgcse-after-reload                                              \
! RUN:     -fgcse-las                                                       \
! RUN:     -fgcse-sm                                                        \
! RUN:     -fipa-cp                                                         \
! RUN:     -finline-functions-called-once                                   \
! RUN:     -fmodulo-sched                                                   \
! RUN:     -fmodulo-sched-allow-regmoves                                    \
! RUN:     -fpeel-loops                                                     \
! RUN:     -frename-registers                                               \
! RUN:     -fschedule-insns2                                                \
! RUN:     -fsingle-precision-constant                                      \
! RUN:     -funsafe-loop-optimizations                                      \
! RUN:     -fuse-linker-plugin                                              \
! RUN:     -fvect-cost-model                                                \
! RUN:     -fvariable-expansion-in-unroller                                 \
! RUN:     -fweb                                                            \
! RUN:     -fwhole-program                                                  \
! RUN:     -fcaller-saves                                                   \
! RUN:     -freorder-blocks                                                 \
! RUN:     -ffat-lto-objects                                                \
! RUN:     -fmerge-constants                                                \
! RUN:     -finline-small-functions                                         \
! RUN:     -ftree-dce                                                       \
! RUN:     -ftree-ter                                                       \
! RUN:     -ftree-vrp                                                       \
! RUN:     -fno-devirtualize 2>&1                                           \
! RUN:     | FileCheck --check-prefix=CHECK-WARNING %s
! CHECK-WARNING-DAG: optimization flag '-finline-limit=1000' is not supported
! CHECK-WARNING-DAG: optimization flag '-finline-limit' is not supported
! CHECK-WARNING-DAG: optimization flag '-fexpensive-optimizations' is not supported
! CHECK-WARNING-DAG: optimization flag '-fno-expensive-optimizations' is not supported
! CHECK-WARNING-DAG: optimization flag '-fno-defer-pop' is not supported
! CHECK-WARNING-DAG: optimization flag '-fkeep-inline-functions' is not supported
! CHECK-WARNING-DAG: optimization flag '-fno-keep-inline-functions' is not supported
! CHECK-WARNING-DAG: optimization flag '-freorder-blocks' is not supported
! CHECK-WARNING-DAG: optimization flag '-ffloat-store' is not supported
! CHECK-WARNING-DAG: optimization flag '-fgcse' is not supported
! CHECK-WARNING-DAG: optimization flag '-fivopts' is not supported
! CHECK-WARNING-DAG: optimization flag '-fprefetch-loop-arrays' is not supported
! CHECK-WARNING-DAG: optimization flag '-fprofile-correction' is not supported
! CHECK-WARNING-DAG: optimization flag '-fprofile-values' is not supported
! CHECK-WARNING-DAG: optimization flag '-fschedule-insns' is not supported
! CHECK-WARNING-DAG: optimization flag '-fsignaling-nans' is not supported
! CHECK-WARNING-DAG: optimization flag '-fstrength-reduce' is not supported
! CHECK-WARNING-DAG: optimization flag '-ftracer' is not supported
! CHECK-WARNING-DAG: optimization flag '-funroll-all-loops' is not supported
! CHECK-WARNING-DAG: optimization flag '-funswitch-loops' is not supported
! CHECK-WARNING-DAG: optimization flag '-falign-labels' is not supported
! CHECK-WARNING-DAG: optimization flag '-falign-labels=100' is not supported
! CHECK-WARNING-DAG: optimization flag '-falign-jumps' is not supported
! CHECK-WARNING-DAG: optimization flag '-falign-jumps=100' is not supported
! CHECK-WARNING-DAG: optimization flag '-fbranch-count-reg' is not supported
! CHECK-WARNING-DAG: optimization flag '-fcaller-saves' is not supported
! CHECK-WARNING-DAG: optimization flag '-fno-default-inline' is not supported
! CHECK-WARNING-DAG: optimization flag '-fgcse-after-reload' is not supported
! CHECK-WARNING-DAG: optimization flag '-fgcse-las' is not supported
! CHECK-WARNING-DAG: optimization flag '-fgcse-sm' is not supported
! CHECK-WARNING-DAG: optimization flag '-fipa-cp' is not supported
! CHECK-WARNING-DAG: optimization flag '-finline-functions-called-once' is not supported
! CHECK-WARNING-DAG: optimization flag '-fmodulo-sched' is not supported
! CHECK-WARNING-DAG: optimization flag '-fmodulo-sched-allow-regmoves' is not supported
! CHECK-WARNING-DAG: optimization flag '-fpeel-loops' is not supported
! CHECK-WARNING-DAG: optimization flag '-frename-registers' is not supported
! CHECK-WARNING-DAG: optimization flag '-fschedule-insns2' is not supported
! CHECK-WARNING-DAG: optimization flag '-fsingle-precision-constant' is not supported
! CHECK-WARNING-DAG: optimization flag '-funsafe-loop-optimizations' is not supported
! CHECK-WARNING-DAG: optimization flag '-fuse-linker-plugin' is not supported
! CHECK-WARNING-DAG: optimization flag '-fvect-cost-model' is not supported
! CHECK-WARNING-DAG: optimization flag '-fvariable-expansion-in-unroller' is not supported
! CHECK-WARNING-DAG: optimization flag '-fweb' is not supported
! CHECK-WARNING-DAG: optimization flag '-fwhole-program' is not supported
! CHECK-WARNING-DAG: optimization flag '-fcaller-saves' is not supported
! CHECK-WARNING-DAG: optimization flag '-freorder-blocks' is not supported
! CHECK-WARNING-DAG: optimization flag '-fmerge-constants' is not supported
! CHECK-WARNING-DAG: optimization flag '-finline-small-functions' is not supported
! CHECK-WARNING-DAG: optimization flag '-ftree-dce' is not supported
! CHECK-WARNING-DAG: optimization flag '-ftree-ter' is not supported
! CHECK-WARNING-DAG: optimization flag '-ftree-vrp' is not supported
! CHECK-WARNING-DAG: optimization flag '-fno-devirtualize' is not supported
