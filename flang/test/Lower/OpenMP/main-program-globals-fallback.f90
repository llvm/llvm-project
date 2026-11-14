! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp -DREGION '-DDIRECTIVE=target map(tofrom:x)' '-DEND_DIRECTIVE=end target' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp -DREGION '-DDIRECTIVE=target parallel map(tofrom:x)' '-DEND_DIRECTIVE=end target parallel' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp -DREGION '-DDIRECTIVE=target data map(tofrom:x)' '-DEND_DIRECTIVE=end target data' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp '-DDIRECTIVE=target enter data map(to:x)' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp '-DDIRECTIVE=target exit data map(delete:x)' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp '-DDIRECTIVE=target update to(x)' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -cpp '-DDIRECTIVE=declare target' -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %t/meta.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -cpp -DDIRECTIVE=barrier -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=GLOBAL
! RUN: %flang_fc1 -fopenmp -emit-hlfir %t/external.f90 -o - | FileCheck %s --check-prefixes=EXTERNAL,GLOBAL
! RUN: %flang_fc1 -fopenmp -fdo-concurrent-to-openmp=device -fopenmp-targets=nvptx64-nvidia-cuda -emit-fir %t/generated-target.f90 -o - | FileCheck %s --check-prefix=DEVICE --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fdo-concurrent-to-openmp=device -fopenmp-is-target-device -fopenmp-targets=nvptx64-nvidia-cuda -emit-fir %t/generated-target.f90 -o - | FileCheck %s --check-prefix=DEVICE --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: bbc -fopenmp -fdo-concurrent-to-openmp=device -emit-fir %t/generated-target.f90 -o - | FileCheck %s --check-prefix=DEVICE --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: %flang_fc1 -fopenmp -fdo-concurrent-to-openmp=host -emit-fir %t/generated-target.f90 -o - | FileCheck %s --check-prefix=HOST
! RUN: %flang_fc1 -fopenmp -cpp -DEARLY_TARGET -DDIRECTIVE=barrier -emit-hlfir %t/internal.f90 -o - | FileCheck %s --check-prefix=KEEP --implicit-check-not='fir.global internal @_QFEx{{[[:space:]]}}'
! RUN: bbc -fopenmp -fdo-concurrent-to-openmp=host -emit-fir %t/generated-target.f90 -o - | FileCheck %s --check-prefix=HOST
! KEEP: fir.alloca i32 {{.*}}bindc_name = "x"
! EXTERNAL-LABEL: func.func @_QQmain()
! EXTERNAL: fir.address_of(@_QFEx)
! EXTERNAL-LABEL: func.func @_QPoutside()
! EXTERNAL: omp.target_data
! GLOBAL: fir.global internal @_QFEx : i32
!--- internal.f90
! Offload before or inside PARALLEL keeps x on the stack; BARRIER alone does not.
program p
  integer :: x,marker
#ifdef EARLY_TARGET
!$omp declare target(marker)
#endif
!$omp parallel
  print *,x
!$omp end parallel
  call worker()
contains
  subroutine worker()
#ifdef REGION
!$omp parallel
#endif
!$omp DIRECTIVE
#ifdef REGION
    x=1
!$omp END_DIRECTIVE
!$omp end parallel
#endif
  end subroutine
end program
!--- meta.f90
! A metadirective selecting NOTHING keeps x on the stack.
program p
  integer :: x
!$omp parallel
  print *,x
!$omp end parallel
!$omp metadirective when(user={condition(.true.)}: nothing)
  x=1
end program
!--- external.f90
! Offload in another external procedure or an interface body must not disable
! global storage selection in MAIN.
program p
  integer :: x
  interface
    subroutine declared()
!$omp declare target
    end subroutine
  end interface
!$omp parallel
  print *,x
!$omp end parallel
end program
subroutine outside()
  integer :: a(4)
!$omp target data map(tofrom:a)
  a=1
!$omp end target data
end subroutine
! Device mapping passes the host tuple to f; host mapping uses global x.
! DEVICE-LABEL: func.func @_QQmain()
! DEVICE: fir.alloca i32 {{.*}}bindc_name = "x"
! DEVICE: %[[HOST_TUPLE:[^ ]+]] = fir.alloca tuple<!fir.ref<i32>>
! DEVICE: %[[TUPLE_MAP:[^ ]+]] = omp.map.info var_ptr(%[[HOST_TUPLE]] : !fir.ref<tuple<!fir.ref<i32>>>
! DEVICE: omp.target
! DEVICE-SAME: map_entries({{.*}}%[[TUPLE_MAP]] -> %[[MAPPED_TUPLE:[^, )]+]]{{[, )]}}
! DEVICE: %[[TUPLE:[^ ]+]] = fir.declare %[[MAPPED_TUPLE]] {{.*}} : (!fir.ref<tuple<!fir.ref<i32>>>)
! DEVICE: fir.call @_QFPf({{[^,]*}}, %[[TUPLE]])
! DEVICE-LABEL: func.func private @_QFPf(
! DEVICE-SAME: !fir.ref<tuple<!fir.ref<i32>>>
! HOST-LABEL: func.func @_QQmain()
! HOST: fir.address_of(@_QFEx)
! HOST: fir.call @_QFPf({{[^,]*}})
! HOST-LABEL: func.func private @_QFPf(
! HOST-SAME: %{{[^: ]+}}: !fir.ref<i32>{{( \{[^{}]*\})?}}) -> i32
! HOST: fir.global internal @_QFEx
!--- generated-target.f90
program p
  implicit none
  integer :: x,i,a(4)
  x=7
  a=0
!$omp parallel num_threads(1)
  print *,x
!$omp end parallel
  do concurrent(i=1:4)
    block
      integer :: v
      v=f(i)
      a(i)=v
    end block
  end do
  call host_use()
contains
  pure integer function f(j)
    integer,intent(in) :: j
    f=j+x
  end function
  subroutine host_use()
    print *,x
  end subroutine
end program
