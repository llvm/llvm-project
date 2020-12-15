!RUN: %f18 -fdebug-dump-symbols -fparse-only %s | FileCheck %s
! Tests for derived type runtime descriptions

module m01
  type :: t1
    integer :: n
  end type
!CHECK: .c.t1, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:0_8 init:[component::component(name=.n.n,genre=1_1,category=0_1,kind=4_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .dt.t1, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t1,sizeinbytes=4_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t1,procptr=NULL(),special=NULL())
!CHECK: .n.n, SAVE, TARGET: ObjectEntity type: CHARACTER(2_8,1) init:"n"
!CHECK: .n.t1, SAVE, TARGET: ObjectEntity type: CHARACTER(2_8,1) init:"t1"
end module

module m02
  type :: parent
    integer :: pn
  end type
  type, extends(parent) :: child
    integer :: cn
  end type
!CHECK: .c.child, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:1_8 init:[component::component(name=.n.cn,genre=1_1,category=0_1,kind=4_1,rank=0_1,offset=4_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=NULL()),component(name=.n.parent,genre=1_1,category=5_1,kind=0_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=.dt.parent,lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .c.parent, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:0_8 init:[component::component(name=.n.pn,genre=1_1,category=0_1,kind=4_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .dt.child, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.child,sizeinbytes=8_8,parent=.dt.parent,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.child,procptr=NULL(),special=NULL())
!CHECK: .dt.parent, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.parent,sizeinbytes=4_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.parent,procptr=NULL(),special=NULL())
end module

module m03
  type :: kpdt(k)
    integer(kind=1), kind :: k = 1
    real(kind=k) :: a
  end type
  type(kpdt(4)) :: x
!CHECK: .c.kpdt.0, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:0_8 init:[component::component(name=.n.a,genre=1_1,category=1_1,kind=4_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .dt.kpdt, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(name=.n.kpdt,parent=NULL(),uninstantiated=NULL(),kindparameter=.kp.kpdt,lenparameterkind=NULL())
!CHECK: .dt.kpdt.0, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.kpdt,sizeinbytes=4_8,parent=NULL(),uninstantiated=.dt.kpdt,kindparameter=.kp.kpdt.0,lenparameterkind=NULL(),component=.c.kpdt.0,procptr=NULL(),special=NULL())
!CHECK: .kp.kpdt, SAVE, TARGET: ObjectEntity type: INTEGER(8) shape: 0_8:0_8 init:[INTEGER(8)::1_8]
!CHECK: .kp.kpdt.0, SAVE, TARGET: ObjectEntity type: INTEGER(8) shape: 0_8:0_8 init:[INTEGER(8)::4_8]
end module

module m04
  type :: tbps
   contains
    procedure :: b2 => s1
    procedure :: b1 => s1
  end type
 contains
  subroutine s1(x)
    class(tbps), intent(in) :: x
  end subroutine
!CHECK: .dt.tbps, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.tbps,name=.n.tbps,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=NULL())
!CHECK: .v.tbps, SAVE, TARGET: ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=s1,name=.n.b1),binding(proc=s1,name=.n.b2)]
end module

module m05
  type :: t
    procedure(s1), pointer :: p1 => s1
  end type
 contains
  subroutine s1(x)
    class(t), intent(in) :: x
  end subroutine
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t,sizeinbytes=24_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=.p.t,special=NULL())
!CHECK: .p.t, SAVE, TARGET: ObjectEntity type: TYPE(procptrcomponent) shape: 0_8:0_8 init:[procptrcomponent::procptrcomponent(name=.n.p1,offset=0_8,initialization=s1)]
end module

module m06
  type :: t
   contains
    procedure :: s1
    generic :: assignment(=) => s1
  end type
  type, extends(t) :: t2
   contains
    procedure :: s1 => s2 ! override
  end type
 contains
  subroutine s1(x, y)
    class(t), intent(out) :: x
    class(t), intent(in) :: y
  end subroutine
  subroutine s2(x, y)
    class(t2), intent(out) :: x
    class(t), intent(in) :: y
  end subroutine
!CHECK: .c.t2, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:0_8 init:[component::component(name=.n.t,genre=1_1,category=5_1,kind=0_1,rank=0_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=.dt.t,lenvalue=NULL(),bounds=NULL(),initialization=NULL())]
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.t,name=.n.t,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.t)
!CHECK: .dt.t2, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.t2,name=.n.t2,sizeinbytes=0_8,parent=.dt.t,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t2,procptr=NULL(),special=NULL())
!CHECK: .s.t, SAVE, TARGET: ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=4_1,rank=0_1,isargdescriptorset=3_1,proc=s1)]
!CHECK: .v.t, SAVE, TARGET: ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=s1,name=.n.s1)]
!CHECK: .v.t2, SAVE, TARGET: ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=s2,name=.n.s1)]
end module

module m07
  type :: t
   contains
    procedure :: s1
    generic :: assignment(=) => s1
  end type
 contains
  impure elemental subroutine s1(x, y)
    class(t), intent(out) :: x
    class(t), intent(in) :: y
  end subroutine
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.t,name=.n.t,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.t)
!CHECK: .s.t, SAVE, TARGET: ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=5_1,rank=0_1,isargdescriptorset=3_1,proc=s1)]
!CHECK: .v.t, SAVE, TARGET: ObjectEntity type: TYPE(binding) shape: 0_8:0_8 init:[binding::binding(proc=s1,name=.n.s1)]
end module

module m08
  type :: t
   contains
    final :: s1, s2, s3
  end type
 contains
  subroutine s1(x)
    type(t) :: x(:)
  end subroutine
  subroutine s2(x)
    type(t) :: x(3,3)
  end subroutine
  impure elemental subroutine s3(x)
    type(t) :: x
  end subroutine
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.t)
!CHECK: .s.t, SAVE, TARGET: ObjectEntity type: TYPE(specialbinding) shape: 0_8:2_8 init:[specialbinding::specialbinding(which=8_1,rank=1_1,isargdescriptorset=1_1,proc=s1),specialbinding(which=8_1,rank=2_1,isargdescriptorset=0_1,proc=s2),specialbinding(which=9_1,rank=0_1,isargdescriptorset=0_1,proc=s3)]
end module

module m09
  type :: t
   contains
    procedure :: rf, ru, wf, wu
    generic :: read(formatted) => rf
    generic :: read(unformatted) => ru
    generic :: write(formatted) => wf
    generic :: write(unformatted) => wu
  end type
 contains
  subroutine rf(x,u,iot,v,iostat,iomsg)
    class(t), intent(inout) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine ru(x,u,iostat,iomsg)
    class(t), intent(inout) :: x
    integer, intent(in) :: u
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine wf(x,u,iot,v,iostat,iomsg)
    class(t), intent(in) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine wu(x,u,iostat,iomsg)
    class(t), intent(in) :: x
    integer, intent(in) :: u
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.t,name=.n.t,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.t)
!CHECK: .s.t, SAVE, TARGET: ObjectEntity type: TYPE(specialbinding) shape: 0_8:3_8 init:[specialbinding::specialbinding(which=16_1,rank=0_1,isargdescriptorset=1_1,proc=rf),specialbinding(which=17_1,rank=0_1,isargdescriptorset=1_1,proc=ru),specialbinding(which=18_1,rank=0_1,isargdescriptorset=1_1,proc=wf),specialbinding(which=19_1,rank=0_1,isargdescriptorset=1_1,proc=wu)]
!CHECK: .v.t, SAVE, TARGET: ObjectEntity type: TYPE(binding) shape: 0_8:3_8 init:[binding::binding(proc=rf,name=.n.rf),binding(proc=ru,name=.n.ru),binding(proc=wf,name=.n.wf),binding(proc=wu,name=.n.wu)]
end module

module m10
  type :: t
  end type
  interface read(formatted)
    procedure :: rf
  end interface
  interface read(unformatted)
    procedure :: ru
  end interface
  interface write(formatted)
    procedure ::wf
  end interface
  interface write(unformatted)
    procedure :: wu
  end interface
 contains
  subroutine rf(x,u,iot,v,iostat,iomsg)
    type(t), intent(inout) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine ru(x,u,iostat,iomsg)
    type(t), intent(inout) :: x
    integer, intent(in) :: u
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine wf(x,u,iot,v,iostat,iomsg)
    type(t), intent(in) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine wu(x,u,iostat,iomsg)
    type(t), intent(in) :: x
    integer, intent(in) :: u
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t,sizeinbytes=0_8,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.t)
!CHECK: .s.t, SAVE, TARGET: ObjectEntity type: TYPE(specialbinding) shape: 0_8:3_8 init:[specialbinding::specialbinding(which=16_1,rank=0_1,isargdescriptorset=0_1,proc=rf),specialbinding(which=17_1,rank=0_1,isargdescriptorset=0_1,proc=ru),specialbinding(which=18_1,rank=0_1,isargdescriptorset=0_1,proc=wf),specialbinding(which=19_1,rank=0_1,isargdescriptorset=0_1,proc=wu)]
end module

module m11
  real, target :: target
  type :: t(len)
    integer(kind=8), len :: len
    real, allocatable :: allocatable(:)
    real, pointer :: pointer => target
    character(len=len) :: chauto
    real :: automatic(len)
  end type
!CHECK: .dt.t, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(name=.n.t,parent=NULL(),uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=.lpk.t)
!CHECK: .lpk.t, SAVE, TARGET: ObjectEntity type: INTEGER(1) shape: 0_8:0_8 init:[INTEGER(1)::8_1]
 contains
  subroutine s1(x)
!CHECK: .b.t.1.allocatable, SAVE, TARGET: ObjectEntity type: TYPE(value) shape: 0_8:1_8,0_8:0_8 init:reshape([value::value(genre=1_1,value=0_8),value(genre=1_1,value=0_8)],shape=[2,1])
!CHECK: .b.t.1.automatic, SAVE, TARGET: ObjectEntity type: TYPE(value) shape: 0_8:1_8,0_8:0_8 init:reshape([value::value(genre=2_1,value=1_8),value(genre=3_1,value=0_8)],shape=[2,1])
!CHECK: .c.t.1, SAVE, TARGET: ObjectEntity type: TYPE(component) shape: 0_8:3_8 init:[component::component(name=.n.allocatable,genre=3_1,category=1_1,kind=4_1,rank=1_1,offset=0_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=.b.t.1.allocatable,initialization=NULL()),component(name=.n.automatic,genre=4_1,category=1_1,kind=4_1,rank=1_1,offset=96_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=.b.t.1.automatic,initialization=NULL()),component(name=.n.chauto,genre=4_1,category=3_1,kind=1_1,rank=0_1,offset=72_8,characterlen=value(genre=3_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=NULL()),component(name=.n.pointer,genre=2_1,category=1_1,kind=4_1,rank=0_1,offset=48_8,characterlen=value(genre=1_1,value=0_8),derived=NULL(),lenvalue=NULL(),bounds=NULL(),initialization=target)]
!CHECK: .dt.t.1, SAVE, TARGET: ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t,sizeinbytes=144_8,parent=NULL(),uninstantiated=.dt.t,kindparameter=NULL(),lenparameterkind=.lpk.t.1,component=.c.t.1,procptr=NULL(),special=NULL())
!CHECK: .lpk.t.1, SAVE, TARGET: ObjectEntity type: INTEGER(1) shape: 0_8:0_8 init:[INTEGER(1)::8_1]
    type(t(*)), intent(in) :: x
  end subroutine
end module
