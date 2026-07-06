! RUN: %python %S/test_modfile.py %s %flang_fc1 -fenumeration-type
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

! Accessibility statement for an enumerator appearing BEFORE the enumeration
! type definition. The enumerator must still be emitted only inside the
! ENUMERATION TYPE block (never as a standalone forward-referencing PARAMETER),
! regardless of the earlier source position of the accessibility statement.
module m6
  private :: green
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end

!Expect: m6.mod
!module m6
!enumeration type::color
!enumerator::red,green,blue
!end enumeration type
!private::green
!end

! Accessibility statement for the enumeration type NAME appearing BEFORE its
! definition (valid Fortran; distinct from the C7116 forward-reference
! prohibition, which only concerns enumeration-type-specs). The type block
! must be emitted correctly and no enumerator may leak out ahead of it.
module m7
  private :: color
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end

!Expect: m7.mod
!module m7
!enumeration type,private::color
!enumerator::red,green,blue
!end enumeration type
!end

