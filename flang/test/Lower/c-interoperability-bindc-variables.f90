! Test lowering of BIND(C) variables
! RUN: bbc -emit-fir %s -o - | FileCheck %s

block data
   integer :: x, y
   common /fortran_name/ x, y
   ! CHECK-LABEL: fir.global common @c_name
   bind(c, name="c_name") /fortran_name/
end block data

module some_module
   ! CHECK-LABEL: fir.global @tomato
  integer, bind(c, name="tomato") :: apple = 42
end module
