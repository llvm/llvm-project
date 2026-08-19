!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  type :: tbps
   contains
    procedure, nopass :: fprocf, fprocs, ffunc1, ffunc2, ffunc3, fsub
   end type
 contains
  function fprocf()
    procedure(), pointer :: fprocf
    fprocf => func
  end
  function fprocs()
    procedure(), pointer :: fprocs
    fprocs => sub
  end
  function ffunc1()
    procedure(), pointer :: ffunc1
    real ffunc1
    ffunc1 => func
  end
  function ffunc2()
    procedure(real), pointer :: ffunc2
    ffunc2 => func
  end
  function ffunc3()
    procedure(func), pointer :: ffunc3
    ffunc3 => func
  end
  function fsub()
    procedure(sub), pointer :: fsub
    fsub => sub
  end
  subroutine sub
  end
  function func()
    func = 0.
  end
end

program p
  use m
  type(tbps) :: x
  procedure(), pointer :: gp
  procedure(real), pointer :: rfp
  procedure(sub), pointer :: sp
  gp => x%fprocf() ! procedure() always ok to assign any procedure to
  gp => x%fprocs()
  gp => x%ffunc1()
  gp => x%ffunc2()
  gp => x%ffunc3()
  gp => x%fsub()
  rfp => x%fprocf() ! can be assigned a procedure() function result
  rfp => x%fprocs() ! can be assigned a procedure() function result
  rfp => x%ffunc1()
  rfp => x%ffunc2()
  rfp => x%ffunc3()
  !ERROR: Procedure pointer 'rfp' associated with result of reference to function 'fsub' that is an incompatible procedure pointer: incompatible procedures: one is a function, the other a subroutine
  rfp => x%fsub()
  sp => x%fprocf() ! can be assigned a procedure() function result
  sp => x%fprocs() ! can be assigned a procedure() function result
  !ERROR: Procedure pointer 'sp' associated with result of reference to function 'ffunc1' that is an incompatible procedure pointer: incompatible procedures: one is a function, the other a subroutine
  sp => x%ffunc1()
  !ERROR: Procedure pointer 'sp' associated with result of reference to function 'ffunc2' that is an incompatible procedure pointer: incompatible procedures: one is a function, the other a subroutine
  sp => x%ffunc2()
  !ERROR: Procedure pointer 'sp' associated with result of reference to function 'ffunc3' that is an incompatible procedure pointer: incompatible procedures: one is a function, the other a subroutine
  sp => x%ffunc3()
  sp => x%fsub()
  !ERROR: Binding 'fprocf' is not a subroutine
  call x%fprocf()
  !ERROR: Binding 'fprocs' is not a subroutine
  call x%fprocs()
  !ERROR: Binding 'ffunc1' is not a subroutine
  call x%ffunc1()
  !ERROR: Binding 'ffunc2' is not a subroutine
  call x%ffunc2()
  !ERROR: Binding 'ffunc3' is not a subroutine
  call x%ffunc3()
  !ERROR: Binding 'fsub' is not a subroutine
  call x%fsub()
end
