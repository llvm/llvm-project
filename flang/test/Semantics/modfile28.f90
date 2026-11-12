! RUN: %python %S/test_modfile.py %s %flang_fc1

! Test UTF-8 support in character literals
! Note: Module files are encoded in UTF-8.

module m
character(kind=4,len=*), parameter :: c4 = 4_"Hi! 你好!"
! In CHARACTER(1) literals, codepoints > 0xff are serialized into UTF-8;
! each of those bytes then gets encoded into UTF-8 for the module file.
character(kind=1,len=*), parameter :: c1 = 1_"Hi! 你好!"
character(kind=4,len=*), parameter :: c4a(*) = [4_"一", 4_"二", 4_"三", 4_"四", 4_"五"]
integer, parameter :: lc4 = len(c4)
integer, parameter :: lc1 = len(c1)
end module m

!Expect: m.mod
!module m
!character(*,4),parameter::c4=4_"Hi! \344\275\240\345\245\275!"
!character(*,1),parameter::c1="Hi! \344\275\240\345\245\275!"
!character(*,4),parameter::c4a(1_8:*)=[CHARACTER(KIND=4,LEN=1)::4_"\344\270\200",4_"\344\272\214",4_"\344\270\211",4_"\345\233\233",4_"\344\272\224"]
!integer(4),parameter::lc4=7_4
!intrinsic::len
!integer(4),parameter::lc1=11_4
!end
