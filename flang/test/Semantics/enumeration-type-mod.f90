! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check correct modfile generation for enumeration types.

! Basic enumeration type
module m1
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type(color) :: c = green
end

!Expect: m1.mod
!module m1
!enumeration type::color
!enumerator::red,green,blue
!end enumeration type
!type(color)::c
!end

! Private enumeration type
module m2
  enumeration type, private :: color
    enumerator :: red, green, blue
  end enumeration type
end

!Expect: m2.mod
!module m2
!enumeration type,private::color
!enumerator::red,green,blue
!end enumeration type
!end

! Multiple enumeration types
module m3
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  enumeration type :: direction
    enumerator :: north, south, east, west
  end enumeration type
end

!Expect: m3.mod
!module m3
!enumeration type::color
!enumerator::red,green,blue
!end enumeration type
!enumeration type::direction
!enumerator::north,south,east,west
!end enumeration type
!end

! Enumeration type with variable declaration
module m4
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type(color) :: default_color = green
  type(color), parameter :: favorite = blue
end

!Expect: m4.mod
!module m4
!enumeration type::color
!enumerator::red,green,blue
!end enumeration type
!type(color)::default_color
!type(color),parameter::favorite=color(3_4)
!end

! USE and re-export
module m5
  use m1, only: color, red, green, blue, c
end

!Expect: m5.mod
!module m5
!use m1,only:color
!use m1,only:red
!use m1,only:green
!use m1,only:blue
!use m1,only:c
!end
