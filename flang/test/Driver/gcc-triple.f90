!! UNSUPPORTED: system-windows

!! Test that --gcc-triple option is working as expected.

! RUN: %flang -v --sysroot=%S/Inputs/fedora_39_tree 2>&1 | FileCheck %s --check-prefix=DEFAULT_TRIPLE
! DEFAULT_TRIPLE: {{^}}Found candidate GCC installation:
! DEFAULT_TRIPLE: fedora_39_tree/usr/lib/gcc/x86_64-linux-gnu/13
! DEFAULT_TRIPLE: {{^}}Found candidate GCC installation:
! DEFAULT_TRIPLE: fedora_39_tree/usr/lib/gcc/x86_64-redhat-linux/13
! DEFAULT_TRIPLE: {{^}}Selected GCC installation:
! DEFAULT_TRIPLE: fedora_39_tree/usr/lib/gcc/x86_64-linux-gnu/13

! RUN: %flang -v --sysroot=%S/Inputs/fedora_39_tree --gcc-triple=x86_64-redhat-linux 2>&1 | FileCheck %s --check-prefix=TRIPLE_EXISTS
! TRIPLE_EXISTS: {{^}}Selected GCC installation:
! TRIPLE_EXISTS: fedora_39_tree/usr/lib/gcc/x86_64-redhat-linux/13

! RUN: %flang -v --sysroot=%S/Inputs/fedora_39_tree --gcc-triple=x86_64-foo-linux 2>&1 | FileCheck %s --check-prefix=TRIPLE_DOES_NOT_EXIST
! TRIPLE_DOES_NOT_EXIST-NOT: x86_64-foo-linux
