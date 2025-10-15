! Test -gsplit-dwarf and -gsplit-dwarf={split,single}.

! RUN: %flang -### -c -target x86_64 -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=SPLIT
! RUN: %flang -### -c -target x86_64 -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefixes=SPLIT
! RUN: %flang -### -c -target x86_64 -gsplit-dwarf=split -g %s 2>&1 | FileCheck %s --check-prefixes=SPLIT

! SPLIT: "-split-dwarf-file" "split-debug.dwo" "-split-dwarf-output" "split-debug.dwo"

! Check warning on non-supported platforms.
! RUN: %flang -### -c -target x86_64-apple-darwin  -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=WARN
! WARN: warning: debug information option '-gsplit-dwarf' is not supported for target 'x86_64-apple-darwin'

! -gno-split-dwarf disables debug fission.
! RUN: %flang -### -c -target x86_64 -gsplit-dwarf -g -gno-split-dwarf %s 2>&1 | FileCheck %s --check-prefix=NOSPLIT
! RUN: %flang -### -c -target x86_64 -gsplit-dwarf=single -g -gno-split-dwarf %s 2>&1 | FileCheck %s --check-prefix=NOSPLIT
! RUN: %flang -### -c -target x86_64 -gno-split-dwarf -g -gsplit-dwarf %s 2>&1 | FileCheck %s --check-prefixes=SPLIT

! NOSPLIT-NOT: "-split-dwarf

! Test -gsplit-dwarf=single.
! RUN: %flang -### -c -target x86_64 -gsplit-dwarf=single -g %s 2>&1 | FileCheck %s --check-prefix=SINGLE

! SINGLE: "-split-dwarf-file" "split-debug.o"
! SINGLE-NOT: "-split-dwarf-output"

! RUN: %flang -### -c -target x86_64 -gsplit-dwarf=single -g -o %tfoo.o %s 2>&1 | FileCheck %s --check-prefix=SINGLE_WITH_FILENAME
! SINGLE_WITH_FILENAME: "-split-dwarf-file" "{{.*}}foo.o"
! SINGLE_WITH_FILENAME-NOT: "-split-dwarf-output"


! Invoke objcopy if not using the integrated assembler.
! RUN: %flang -### -c -target x86_64-unknown-linux-gnu -fno-integrated-as -gsplit-dwarf -g %s 2>&1 | FileCheck %s --check-prefix=OBJCOPY
! OBJCOPY:      objcopy{{(.exe)?}}
! OBJCOPY-SAME: --extract-dwo
! OBJCOPY-NEXT: objcopy{{(.exe)?}}
! OBJCOPY-SAME: --strip-dwo

! RUN: not %flang -target powerpc-ibm-aix -gdwarf-4 -gsplit-dwarf %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=UNSUP_OPT_AIX
! RUN: not %flang -target powerpc64-ibm-aix -gdwarf-4 -gsplit-dwarf %s 2>&1 \
! RUN: | FileCheck %s --check-prefix=UNSUP_OPT_AIX64

! UNSUP_OPT_AIX: error: unsupported option '-gsplit-dwarf' for target 'powerpc-ibm-aix'
! UNSUP_OPT_AIX64: error: unsupported option '-gsplit-dwarf' for target 'powerpc64-ibm-aix'
