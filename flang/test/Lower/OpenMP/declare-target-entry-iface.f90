!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

! Test that a bare '!$omp declare target' inside multiple entry procedures and
! interfaces marks all impacted functions as declare-target, including when
! complex multi-level nests of subroutines, contained subroutines and
! interfaces are present.

! CHECK: func.func {{.*}}@{{.*}}top()
! CHECK-NOT: omp.declare_target
subroutine top()
  interface
    subroutine topiface()
      interface
        subroutine topifacenested()
        end subroutine
        subroutine topifacenesteddt()
          !$omp declare target
        end subroutine
      end interface
    end subroutine
    subroutine topifacedt()
      !$omp declare target
      interface
        subroutine topifacedtnested()
        end subroutine
        subroutine topifacedtnesteddt()
          !$omp declare target
        end subroutine
      end interface
    end subroutine
  end interface

  call topiface()
  call topifacedt()
  call topifacenested()
  call topifacenesteddt()
  call topifacedtnested()
  call topifacedtnesteddt()
  call topnested()
  call topnesteddt()

  contains
    ! CHECK: func.func {{.*}}@{{.*}}topnested()
    ! CHECK-NOT: omp.declare_target
    subroutine topnested()
      interface
        subroutine topnestediface()
        end subroutine
        subroutine topnestedifacedt()
          !$omp declare target
        end subroutine
      end interface

      call topnestediface()
      call topnestedifacedt()
    end subroutine
    ! CHECK: func.func {{.*}}@{{.*}}topnesteddt()
    ! CHECK-SAME: omp.declare_target
    subroutine topnesteddt()
      !$omp declare target
      interface
        subroutine topnesteddtiface()
        end subroutine
        subroutine topnesteddtifacedt()
          !$omp declare target
        end subroutine
      end interface

      call topnesteddtiface()
      call topnesteddtifacedt()
    end subroutine
end subroutine

! CHECK: func.func {{.*}}@{{.*}}topdt()
! CHECK-SAME: omp.declare_target
subroutine topdt()
  !$omp declare target
  interface
    subroutine topdtiface()
      interface
        subroutine topdtifacenested()
        end subroutine
        subroutine topdtifacenesteddt()
          !$omp declare target
        end subroutine
      end interface
    end subroutine
    subroutine topdtifacedt()
      !$omp declare target
      interface
        subroutine topdtifacedtnested()
        end subroutine
        subroutine topdtifacedtnesteddt()
          !$omp declare target
        end subroutine
      end interface
    end subroutine
  end interface

  call topdtiface()
  call topdtifacedt()
  call topdtifacenested()
  call topdtifacenesteddt()
  call topdtifacedtnested()
  call topdtifacedtnesteddt()
  call topdtnested()
  call topdtnesteddt()

  contains
    ! CHECK: func.func {{.*}}@{{.*}}topdtnested()
    ! CHECK-NOT: omp.declare_target
    subroutine topdtnested()
      interface
        subroutine topdtnestediface()
        end subroutine
        subroutine topdtnestedifacedt()
          !$omp declare target
        end subroutine
      end interface

      call topdtnestediface()
      call topdtnestedifacedt()
    end subroutine
    ! CHECK: func.func {{.*}}@{{.*}}topdtnesteddt()
    ! CHECK-SAME: omp.declare_target
    subroutine topdtnesteddt()
      !$omp declare target
      interface
        subroutine topdtnesteddtiface()
        end subroutine
        subroutine topdtnesteddtifacedt()
          !$omp declare target
        end subroutine
      end interface

      call topdtnesteddtiface()
      call topdtnesteddtifacedt()
    end subroutine
end subroutine

! CHECK: func.func {{.*}}@{{.*}}split()
! CHECK-NOT: omp.declare_target
subroutine split()

  interface
    subroutine splitiface()
    end subroutine
    subroutine splitifacedt()
      !$omp declare target
    end subroutine
  end interface

  call splitiface()
  call splitifacedt()
  return

! CHECK: func.func {{.*}}@{{.*}}splita()
! CHECK-NOT: omp.declare_target
entry splita()
  return

! CHECK: func.func {{.*}}@{{.*}}splitb()
! CHECK-NOT: omp.declare_target
entry splitb()
  return

  contains
    ! CHECK: func.func {{.*}}@{{.*}}splitnested()
    ! CHECK-NOT: omp.declare_target
    subroutine splitnested()
    end subroutine
    ! CHECK: func.func {{.*}}@{{.*}}splitnesteddt()
    ! CHECK-SAME: omp.declare_target
    subroutine splitnesteddt()
      !$omp declare target
    end subroutine
end subroutine

! CHECK: func.func {{.*}}@{{.*}}splitdt()
! CHECK-SAME: omp.declare_target
subroutine splitdt()
  !$omp declare target

  interface
    subroutine splitdtiface()
    end subroutine
    subroutine splitdtifacedt()
      !$omp declare target
    end subroutine
  end interface

  call splitdtiface()
  call splitdtifacedt()
  call splitdtnested()
  call splitdtnesteddt()
  return

! CHECK: func.func {{.*}}@{{.*}}splitdta()
! CHECK-SAME: omp.declare_target
entry splitdta()
  return

! CHECK: func.func {{.*}}@{{.*}}splitdtb()
! CHECK-SAME: omp.declare_target
entry splitdtb()
  return

  contains
    ! CHECK: func.func {{.*}}@{{.*}}splitdtnested()
    ! CHECK-NOT: omp.declare_target
    subroutine splitdtnested()
    end subroutine
    ! CHECK: func.func {{.*}}@{{.*}}splitdtnesteddt()
    ! CHECK-SAME: omp.declare_target
    subroutine splitdtnesteddt()
      !$omp declare target
    end subroutine
end subroutine

! CHECK: func.func {{.*}}@{{.*}}main()
! CHECK-NOT: omp.declare_target
program main
  interface
    subroutine progiface()
    end subroutine
    subroutine progifacedt()
      !$omp declare target
    end subroutine
  end interface

  call progiface()
  call progifacedt()
end program

! CHECK: func.func {{.*}}@{{.*}}topiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}topifacenested()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topifacenesteddt()
! CHECK-SAME: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topifacedtnested()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topifacedtnesteddt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}topnestediface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topnestedifacedt()
! CHECK-SAME: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topnesteddtiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topnesteddtifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}topdtiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}topdtifacenested()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtifacenesteddt()
! CHECK-SAME: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtifacedtnested()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtifacedtnesteddt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}topdtnestediface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtnestedifacedt()
! CHECK-SAME: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtnesteddtiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}topdtnesteddtifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}splitiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}splitifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}splitdtiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}splitdtifacedt()
! CHECK-SAME: omp.declare_target

! CHECK: func.func {{.*}}@{{.*}}progiface()
! CHECK-NOT: omp.declare_target
! CHECK: func.func {{.*}}@{{.*}}progifacedt()
! CHECK-SAME: omp.declare_target
