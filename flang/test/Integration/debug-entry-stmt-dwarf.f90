! host_sub and alt_entry are two entry points of the same subroutine. Check
! DW_AT_decl_line reports each one on its own source line: the SUBROUTINE line
! for host_sub, the ENTRY line for alt_entry, at -O0 through -O3.

! REQUIRES: x86-registered-target

! RUN: %flang -g -O0 --target=x86_64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
! RUN: %flang -g -O1 --target=x86_64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
! RUN: %flang -g -O2 --target=x86_64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s
! RUN: %flang -g -O3 --target=x86_64-unknown-linux-gnu -c %s -o %t.o
! RUN: llvm-dwarfdump --debug-info %t.o | FileCheck %s


subroutine host_sub(i)
    integer :: i
    i = 1
    return
entry alt_entry(i)
    i = 2
end subroutine

! CHECK: DW_TAG_subprogram
! CHECK-NOT: DW_TAG
! CHECK: DW_AT_name ("host_sub")
! CHECK-NOT: DW_TAG
! CHECK: DW_AT_decl_line (17)

! CHECK: DW_TAG_subprogram
! CHECK-NOT: DW_TAG
! CHECK: DW_AT_name ("alt_entry")
! CHECK-NOT: DW_TAG
! CHECK: DW_AT_decl_line (21)
