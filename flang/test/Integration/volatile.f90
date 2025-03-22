! RUN: bbc %s -o - | FileCheck %s
logical, volatile :: a
logical           :: b
integer           :: i
a = .false.
b = a
a = .true.
end

! CHECK: %{{.+}} = fir.load %{{.+}} : !fir.ref<!fir.logical<4>, volatile>
! CHECK: hlfir.assign %{{.+}} to %{{.+}} : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
