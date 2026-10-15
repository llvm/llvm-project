! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 PROTECTED_TARGET in procedure interfaces.  J3/26-007r1 does not list
! the attribute among the characteristics of a dummy data object (15.3.2.2)
! or of a function result (15.3.3), so interfaces that differ only in it are
! compatible in either direction.  An actual argument is checked against the
! interface used in the reference (C872).  A difference in CONTIGUOUS, which
! is a characteristic, is still diagnosed.
module m1
  real, pointer, protected_target :: pa(:)
  abstract interface
    subroutine iface_pt(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    subroutine iface_plain(a)
      real, pointer, intent(in) :: a(:)
    end
    subroutine iface_contig(a)
      real, pointer, contiguous, intent(in) :: a(:)
    end
    function iface_pt_result()
      real, pointer, protected_target :: iface_pt_result(:)
    end
    function iface_plain_result()
      real, pointer :: iface_plain_result(:)
    end
    function iface_contig_result()
      real, pointer, contiguous :: iface_contig_result(:)
    end
  end interface
  abstract interface
    subroutine iface_pt_cb(cb)
      import iface_pt
      procedure(iface_pt) :: cb
    end
    subroutine iface_plain_cb(cb)
      import iface_plain
      procedure(iface_plain) :: cb
    end
    subroutine iface_contig_cb(cb)
      import iface_contig
      procedure(iface_contig) :: cb
    end
  end interface
 contains
  subroutine sub_pt(a)
    real, pointer, protected_target, intent(in) :: a(:)
  end
  subroutine sub_plain(a)
    real, pointer, intent(in) :: a(:)
  end
  function fun_pt()
    real, pointer, protected_target :: fun_pt(:)
    fun_pt => null()
  end
  function fun_plain()
    real, pointer :: fun_plain(:)
    fun_plain => null()
  end
  subroutine sub_pt_cb(cb)
    procedure(iface_pt) :: cb
  end
  subroutine sub_plain_cb(cb)
    procedure(iface_plain) :: cb
  end
  subroutine take_pt(proc)
    procedure(iface_pt) :: proc
    call proc(pa)
  end
  subroutine take_plain(proc)
    procedure(iface_plain) :: proc
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pa'
    call proc(pa)
  end
  subroutine take_contig(proc)
    procedure(iface_contig) :: proc
  end
  subroutine take_pt_result(proc)
    procedure(iface_pt_result) :: proc
  end
  subroutine take_plain_result(proc)
    procedure(iface_plain_result) :: proc
  end
  subroutine take_pt_cb(proc)
    procedure(iface_pt_cb) :: proc
  end
  subroutine take_plain_cb(proc)
    procedure(iface_plain_cb) :: proc
  end
  subroutine take_contig_cb(proc)
    procedure(iface_contig_cb) :: proc
  end
  subroutine pointers
    procedure(iface_pt), pointer :: p1
    procedure(iface_plain), pointer :: p2
    procedure(iface_contig), pointer :: p3
    procedure(iface_pt_result), pointer :: p4
    procedure(iface_plain_result), pointer :: p5
    procedure(iface_contig_result), pointer :: p6
    procedure(iface_pt_cb), pointer :: p7
    procedure(iface_plain_cb), pointer :: p8
    p1 => sub_pt
    p1 => sub_plain
    p2 => sub_plain
    p2 => sub_pt
    !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'sub_plain': incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous
    p3 => sub_plain
    !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'sub_pt': incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous
    p3 => sub_pt
    p4 => fun_pt
    p4 => fun_plain
    p5 => fun_plain
    p5 => fun_pt
    !ERROR: Function pointer 'p6' associated with incompatible function designator 'fun_pt': function results have incompatible attributes
    p6 => fun_pt
    p7 => sub_pt_cb
    p7 => sub_plain_cb
    p8 => sub_plain_cb
    p8 => sub_pt_cb
    ! The interface of the pointer, not that of its target, applies.
    p1 => sub_plain
    call p1(pa)
    p2 => sub_pt
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pa'
    call p2(pa)
  end
  subroutine dummy_procedures
    call take_pt(sub_pt)
    call take_pt(sub_plain)
    call take_plain(sub_plain)
    call take_plain(sub_pt)
    !ERROR: Actual procedure argument has interface incompatible with dummy argument 'proc=': incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous
    call take_contig(sub_plain)
    call take_pt_result(fun_plain)
    call take_plain_result(fun_pt)
    call take_pt_cb(sub_plain_cb)
    call take_plain_cb(sub_pt_cb)
    !ERROR: Actual procedure argument has interface incompatible with dummy argument 'proc=': incompatible dummy argument #1: incompatible dummy procedure interfaces: incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous
    call take_contig_cb(sub_pt_cb)
  end
end

module m2
  use m1
  interface
    module subroutine dummy_added(a)
      real, pointer, intent(in) :: a(:)
    end
    module subroutine dummy_dropped(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    module subroutine contig_dropped(a)
      real, pointer, contiguous, protected_target, intent(in) :: a(:)
    end
    module function result_added()
      real, pointer :: result_added(:)
    end
    module function result_dropped()
      real, pointer, protected_target :: result_dropped(:)
    end
    module subroutine cb_dummy_added(cb)
      procedure(iface_plain) :: cb
    end
    module subroutine cb_dummy_dropped(cb)
      procedure(iface_pt) :: cb
    end
    module subroutine cb_contig_dropped(cb)
      procedure(iface_contig) :: cb
    end
    module function ptr_result_dummy_added() result(r)
      procedure(iface_plain), pointer :: r
    end
    module function ptr_result_dummy_dropped() result(r)
      procedure(iface_pt), pointer :: r
    end
  end interface
end

submodule(m2) sm2
 contains
  module subroutine dummy_added(a)
    real, pointer, protected_target, intent(in) :: a(:)
  end
  module subroutine dummy_dropped(a)
    real, pointer, intent(in) :: a(:)
  end
  module subroutine contig_dropped(a)
    !ERROR: Dummy argument 'a' does not have the CONTIGUOUS attribute; the corresponding argument in the interface body does
    real, pointer, intent(in) :: a(:)
  end
  module function result_added()
    real, pointer, protected_target :: result_added(:)
    result_added => null()
  end
  module function result_dropped()
    real, pointer :: result_dropped(:)
    result_dropped => null()
  end
  module subroutine cb_dummy_added(cb)
    procedure(iface_pt) :: cb
  end
  module subroutine cb_dummy_dropped(cb)
    procedure(iface_plain) :: cb
  end
  module subroutine cb_contig_dropped(cb)
    !ERROR: Dummy procedure 'cb' is not compatible with the corresponding argument in the interface body: incompatible dummy procedure interfaces: incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous
    procedure(iface_plain) :: cb
  end
  module function ptr_result_dummy_added() result(r)
    procedure(iface_pt), pointer :: r
    r => null()
  end
  module function ptr_result_dummy_dropped() result(r)
    procedure(iface_plain), pointer :: r
    r => null()
  end
end

module m3
  use m1, only: pa
  type base
   contains
    procedure, nopass :: dummy_added => sub_plain
    procedure, nopass :: dummy_dropped => sub_pt
    procedure, nopass :: contig_dropped => sub_contig
    procedure, nopass :: result_added => fun_plain
    procedure, nopass :: result_dropped => fun_pt
    procedure :: pass_dummy_added => base_plain
    procedure :: pass_dummy_dropped => base_pt
    procedure :: pass_contig_dropped => base_contig
  end type
  type, extends(base) :: ext
   contains
    procedure, nopass :: dummy_added => sub_pt
    procedure, nopass :: dummy_dropped => sub_plain
    !ERROR: A NOPASS type-bound procedure and its override must have identical interfaces
    procedure, nopass :: contig_dropped => sub_plain
    procedure, nopass :: result_added => fun_pt
    procedure, nopass :: result_dropped => fun_plain
    procedure :: pass_dummy_added => ext_pt
    procedure :: pass_dummy_dropped => ext_plain
    !ERROR: A type-bound procedure and its override must have compatible interfaces
    procedure :: pass_contig_dropped => ext_plain
  end type
 contains
  subroutine sub_pt(a)
    real, pointer, protected_target, intent(in) :: a(:)
  end
  subroutine sub_plain(a)
    real, pointer, intent(in) :: a(:)
  end
  subroutine sub_contig(a)
    real, pointer, contiguous, intent(in) :: a(:)
  end
  function fun_pt()
    real, pointer, protected_target :: fun_pt(:)
    fun_pt => null()
  end
  function fun_plain()
    real, pointer :: fun_plain(:)
    fun_plain => null()
  end
  subroutine base_pt(this, a)
    class(base), intent(in) :: this
    real, pointer, protected_target, intent(in) :: a(:)
  end
  subroutine base_plain(this, a)
    class(base), intent(in) :: this
    real, pointer, intent(in) :: a(:)
  end
  subroutine base_contig(this, a)
    class(base), intent(in) :: this
    real, pointer, contiguous, intent(in) :: a(:)
  end
  subroutine ext_pt(this, a)
    class(ext), intent(in) :: this
    real, pointer, protected_target, intent(in) :: a(:)
  end
  subroutine ext_plain(this, a)
    class(ext), intent(in) :: this
    real, pointer, intent(in) :: a(:)
  end
  ! The binding of the declared type applies.
  subroutine calls(x, y)
    class(base), intent(in) :: x
    type(ext), intent(in) :: y
    call x%dummy_dropped(pa)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pa'
    call y%dummy_dropped(pa)
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pa'
    call x%dummy_added(pa)
    call y%dummy_added(pa)
  end
end

module m4
  use m2
 contains
  ! The interface body of a separate module procedure applies.
  subroutine calls
    !ERROR: POINTER dummy argument 'a=' must have the PROTECTED_TARGET attribute to be associated with the target of PROTECTED_TARGET pointer 'pa'
    call dummy_added(pa)
    call dummy_dropped(pa)
  end
end

! Procedures passed to procedures with implicit interfaces
subroutine implicit_references
  use m3, only: sub_pt, sub_plain, sub_contig
  call zpt(sub_pt)
  call zpt(sub_plain)
  call zcontig(sub_plain)
  !WARNING: Reference to the procedure 'zcontig' has an implicit interface that is distinct from another reference: incompatible dummy argument #1: incompatible dummy procedure interfaces: incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous [-Wincompatible-implicit-interfaces]
  call zcontig(sub_contig)
end

! External procedures and interfaces to them
subroutine xpt(a)
  real, pointer, protected_target, intent(in) :: a(:)
end
subroutine xplain(a)
  real, pointer, intent(in) :: a(:)
end
subroutine xcontig(a)
  real, pointer, contiguous, intent(in) :: a(:)
end

subroutine externals1
  interface
    subroutine xpt(a)
      real, pointer, intent(in) :: a(:)
    end
    subroutine xplain(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    !WARNING: The global subprogram 'xcontig' is not compatible with its local procedure declaration (incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous) [-Wexternal-interface-mismatch]
    subroutine xcontig(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    subroutine yplain(a)
      real, pointer, intent(in) :: a(:)
    end
    subroutine ycontig(a)
      real, pointer, intent(in) :: a(:)
    end
  end interface
end

subroutine externals2
  interface
    subroutine yplain(a)
      real, pointer, protected_target, intent(in) :: a(:)
    end
    !WARNING: The external interface 'ycontig' is not compatible with an earlier definition (incompatible dummy argument #1: incompatible dummy data object attributes: Contiguous) [-Wexternal-interface-mismatch]
    subroutine ycontig(a)
      real, pointer, contiguous, protected_target, intent(in) :: a(:)
    end
  end interface
end
