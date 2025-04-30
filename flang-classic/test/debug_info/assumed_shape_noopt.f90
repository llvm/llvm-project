!RUN: %flang -g -S -emit-llvm %s -o - | llc -O0 -fast-isel=false -global-isel=false -filetype=obj -o %t
!RUN: llvm-dwarfdump %t | FileCheck %s

!CHECK-LABEL: DW_TAG_subprogram
!COM: make sure DLOC's DW_AT_location is available
!CHECK-LABEL: DW_TAG_subprogram
  !CHECK: DW_AT_name      ("show")
    !CHECK:[[DLOC:0x[0-9a-f]+]]: DW_TAG_formal_parameter
      !CHECK: DW_AT_location
    !CHECK:[[ARRAY:0x[0-9a-f]+]]: DW_TAG_formal_parameter
      !CHECK: DW_AT_location
      !CHECK: DW_AT_type ([[TYPE:0x[0-9a-f]+]]
    !CHECK: [[TYPE]]: DW_TAG_array_type
      !CHECK: DW_AT_data_location ([[DLOC]])

subroutine show (array)
  integer :: array(:,:)

  print *, array
end subroutine show

program test
  interface
     subroutine show (array)
       integer :: array(:,:)
     end subroutine show
  end interface

  integer :: parray(4,4) = reshape((/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16/),(/4,4/))

  call show (parray(1:2,1:2))
end program test
