! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPrandom_test
subroutine random_test
    ! CHECK-DAG: [[ss:%[0-9]+]] = fir.alloca {{.*}}random_testEss
    ! CHECK-DAG: [[vv:%[0-9]+]] = fir.alloca {{.*}}random_testEvv
    integer ss, vv(40)
    ! CHECK-DAG: [[rr:%[0-9]+]] = fir.alloca {{.*}}random_testErr
    ! CHECK-DAG: [[aa:%[0-9]+]] = fir.alloca {{.*}}random_testEaa
    real rr, aa(5)
    ! CHECK: fir.call @_FortranARandomInit(%true{{.*}}, %false{{.*}}) : (i1, i1) -> none
    call random_init(.true., .false.)
    ! CHECK: [[box:%[0-9]+]] = fir.embox [[ss]]
    ! CHECK: [[argbox:%[0-9]+]] = fir.convert [[box]]
    ! CHECK: fir.call @_FortranARandomSeedSize([[argbox]]
    call random_seed(size=ss)
    print*, 'size: ', ss
    ! CHECK: fir.call @_FortranARandomSeedDefaultPut() : () -> none
    call random_seed()
    ! CHECK: [[box:%[0-9]+]] = fir.embox [[rr]]
    ! CHECK: [[argbox:%[0-9]+]] = fir.convert [[box]]
    ! CHECK: fir.call @_FortranARandomNumber([[argbox]]
    call random_number(rr)
    print*, rr
    ! CHECK: [[box:%[0-9]+]] = fir.embox [[vv]]
    ! CHECK: [[argbox:%[0-9]+]] = fir.convert [[box]]
    ! CHECK: fir.call @_FortranARandomSeedGet([[argbox]]
    call random_seed(get=vv)
  ! print*, 'get:  ', vv(1:ss)
    ! CHECK: [[box:%[0-9]+]] = fir.embox [[vv]]
    ! CHECK: [[argbox:%[0-9]+]] = fir.convert [[box]]
    ! CHECK: fir.call @_FortranARandomSeedPut([[argbox]]
    call random_seed(put=vv)
    print*, 'put:  ', vv(1:ss)
    ! CHECK: [[box:%[0-9]+]] = fir.embox [[aa]]
    ! CHECK: [[argbox:%[0-9]+]] = fir.convert [[box]]
    ! CHECK: fir.call @_FortranARandomNumber([[argbox]]
    call random_number(aa)
    print*, aa
  end
  