!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
module m
  interface
    integer function ifunc(n)
      integer, intent(in) :: n
    end function
    real function rfunc(x)
      real, intent(in) :: x
    end function
  end interface
  external extrfunc
  real extrfunc
  type :: t1(kind,len)
    integer(kind=1), kind :: kind = 4
    integer(kind=2), len :: len = 1
    integer(kind=kind) :: j
    real(kind=kind) :: x(2,2)
    complex(kind=kind) :: z
    logical(kind=kind) :: t
    character(kind=5-kind) :: c(2)
    real(kind=kind), pointer :: xp(:,:)
    procedure(ifunc), pointer, nopass :: ifptr
    procedure(rfunc), pointer, nopass :: rp
    procedure(real), pointer, nopass :: xrp
  end type
 contains
  subroutine s1
    procedure(ifunc), pointer :: ifptr ! CHECK: ifptr, EXTERNAL, POINTER (Function, InDataStmt) size=24 offset=0: ProcEntity ifunc => ifunc
    data ifptr/ifunc/
  end subroutine
  subroutine s2
    integer(kind=1) :: j1 ! CHECK: j1 (InDataStmt) size=1 offset=0: ObjectEntity type: INTEGER(1) init:66_1
    data j1/66/
  end subroutine
  subroutine s3
    integer :: jd ! CHECK: jd (InDataStmt) size=4 offset=0: ObjectEntity type: INTEGER(4) init:666_4
    data jd/666/
  end subroutine
  subroutine s4
    logical :: lv(2) ! CHECK: lv (InDataStmt) size=8 offset=0: ObjectEntity type: LOGICAL(4) shape: 1_8:2_8 init:[LOGICAL(4)::.false._4,.true._4]
    data lv(1)/.false./
    data lv(2)/.true./
  end subroutine
  subroutine s5
    real :: rm(2,2) ! CHECK: rm (InDataStmt) size=16 offset=0: ObjectEntity type: REAL(4) shape: 1_8:2_8,1_8:2_8 init:reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2])
    data rm/1,2,3,4/
  end subroutine
  subroutine s6
    character(len=8) :: ssd ! CHECK: ssd (InDataStmt) size=8 offset=0: ObjectEntity type: CHARACTER(8_4,1) init:"abcdefgh"
    data ssd(1:4)/'abcd'/,ssd(5:8)/'efgh'/
  end subroutine
  subroutine s7
    complex(kind=16) :: zv(-1:1) ! CHECK: zv (InDataStmt) size=96 offset=0: ObjectEntity type: COMPLEX(16) shape: -1_8:1_8 init:[COMPLEX(16)::(1._16,2._16),(3._16,4._16),(5._16,6._16)]
    data (zv(j), j=1,0,-1)/(5,6),(3,4)/
    data (zv(j)%im, zv(j)%re, j=-1,-1,-9)/2,1/
  end subroutine
  real function rfunc2(x)
    real, intent(in) :: x
    rfunc2 = x + 1.
  end function
  subroutine s8
    procedure(rfunc), pointer :: rfptr ! CHECK: rfptr, EXTERNAL, POINTER (Function, InDataStmt) size=24 offset=0: ProcEntity rfunc => rfunc2
    data rfptr/rfunc2/
  end subroutine
  subroutine s10
    real, target, save :: arr(3,4) ! CHECK: arr, SAVE, TARGET size=48 offset=0: ObjectEntity type: REAL(4) shape: 1_8:3_8,1_8:4_8
    real, pointer :: xpp(:,:) ! CHECK: xpp, POINTER (InDataStmt) size=72 offset=48: ObjectEntity type: REAL(4) shape: :,: init:arr
    data xpp/arr/
  end subroutine
  integer function ifunc2(n)
    integer, intent(in) :: n
    ifunc2 = n + 1
  end function
  subroutine s11
    real, target, save :: arr(3,4) ! CHECK: arr, SAVE, TARGET size=48 offset=0: ObjectEntity type: REAL(4) shape: 1_8:3_8,1_8:4_8
    type(t1) :: d1 = t1(1,reshape([1,2,3,4],[2,2]),(6.,7.),.false.,'ab',arr,ifunc2,rfunc,extrfunc) ! CHECK: d1 size=184 offset=48: ObjectEntity type: TYPE(t1(kind=4_1,len=1_2)) init:t1(kind=4_1,len=1_2)(j=1_4,x=reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2]),z=(6._4,7._4),t=.false._4,c=[CHARACTER(KIND=1,LEN=1)::"a","a"],xp=arr,ifptr=ifunc2,rp=rfunc,xrp=extrfunc)
    type(t1(4,len=1)) :: d2 = t1(4)(xrp=extrfunc,rp=rfunc,ifptr=ifunc2,xp=arr,c='a&
      &b',t=.false.,z=(6.,7.),x=reshape([1,2,3,4],[2,2]),j=1) ! CHECK: d2 size=184 offset=232: ObjectEntity type: TYPE(t1(kind=4_1,len=1_2)) init:t1(kind=4_1,len=1_2)(j=1_4,x=reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2]),z=(6._4,7._4),t=.false._4,c=[CHARACTER(KIND=1,LEN=1)::"a","a"],xp=arr,ifptr=ifunc2,rp=rfunc,xrp=extrfunc)
    type(t1(2+2)) :: d3 ! CHECK: d3 (InDataStmt) size=184 offset=416: ObjectEntity type: TYPE(t1(kind=4_1,len=1_2)) init:t1(kind=4_1,len=1_2)(j=1_4,x=reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2]),z=(6._4,7._4),t=.false._4,c=[CHARACTER(KIND=1,LEN=1)::"a","a"],xp=arr,ifptr=ifunc2,rp=rfunc,xrp=extrfunc)
    data d3/t1(1,reshape([1,2,3,4],[2,2]),(6.,7.),.false.,'ab',arr,ifunc2,rfunc,extrfunc)/
    type(t1) :: d4 ! CHECK: d4 (InDataStmt) size=184 offset=600: ObjectEntity type: TYPE(t1(kind=4_1,len=1_2)) init:t1(kind=4_1,len=1_2)(j=1_4,x=reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2]),z=(6._4,7._4),t=.false._4,c=[CHARACTER(KIND=1,LEN=1)::"a","a"],xp=arr,ifptr=ifunc2,rp=rfunc,xrp=extrfunc)
    data d4/t1(4)(xrp=extrfunc,rp=rfunc,ifptr=ifunc2,xp=arr,c='ab',t=.false.,z=(6&
      &.,7.),x=reshape([1,2,3,4],[2,2]),j=1)/
    type(t1) :: d5 ! CHECK: d5 (InDataStmt) size=184 offset=784: ObjectEntity type: TYPE(t1(kind=4_1,len=1_2)) init:t1(kind=4_1,len=1_2)(j=1_4,x=reshape([REAL(4)::1._4,2._4,3._4,4._4],shape=[2,2]),z=(6._4,7._4),t=.false._4,c=[CHARACTER(KIND=1,LEN=1)::"a","b"],xp=arr,ifptr=ifunc2,rp=rfunc,xrp=extrfunc)
    data d5%j/1/,d5%x/1,2,3,4/,d5%z%re/6./,d5%z%im/7./,d5%t/.false./,d5%c(1:1)/'a'/,d5%c(2:&
      &2)/'b'/,d5%xp/arr/,d5%ifptr/ifunc2/,d5%rp/rfunc/,d5%xrp/extrfunc/
  end subroutine
  subroutine s12
    procedure(rfunc), pointer :: pp ! CHECK: pp, EXTERNAL, POINTER (Function, InDataStmt) size=24 offset=0: ProcEntity rfunc => rfunc2
    data pp/rfunc2/
  end subroutine
end module
