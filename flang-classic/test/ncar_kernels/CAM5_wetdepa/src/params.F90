module params

   use kinds_mod
   use shr_const_mod
   integer, public, parameter :: pcols=16
   integer, public, parameter :: pver=30
   real(r8), parameter :: gravit = shr_const_g 
   real(r8), parameter :: tmelt = shr_const_tkfrz
   real(r8), parameter :: rair = shr_const_rdair
   character(len=4), parameter :: cam_physpkg_is = 'cam5'

end module params
