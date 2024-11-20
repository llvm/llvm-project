! This test checks lowering of OpenACC device_type clause on directive where its
! position and the clauses that follow have special semantic

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine sub1()

  !$acc parallel num_workers(16)
  !$acc end parallel

! CHECK: acc.parallel num_workers(%c16{{.*}} : i32) {

  !$acc parallel num_workers(1) device_type(nvidia) num_workers(16)
  !$acc end parallel

! CHECK: acc.parallel num_workers(%c1{{.*}} : i32, %c16{{.*}} : i32 [#acc.device_type<nvidia>])

  !$acc parallel device_type(*) num_workers(1) device_type(nvidia) num_workers(16)
  !$acc end parallel

! CHECK: acc.parallel num_workers(%c1{{.*}} : i32 [#acc.device_type<star>], %c16{{.*}} : i32 [#acc.device_type<nvidia>])

  !$acc parallel vector_length(1)
  !$acc end parallel

! CHECK: acc.parallel vector_length(%c1{{.*}} : i32)

  !$acc parallel device_type(multicore) vector_length(1)
  !$acc end parallel

! CHECK: acc.parallel vector_length(%c1{{.*}} : i32 [#acc.device_type<multicore>])

  !$acc parallel num_gangs(2) device_type(nvidia) num_gangs(4)
  !$acc end parallel

! CHECK: acc.parallel num_gangs({%c2{{.*}} : i32}, {%c4{{.*}} : i32} [#acc.device_type<nvidia>])

  !$acc parallel num_gangs(2) device_type(nvidia) num_gangs(1, 1, 1)
  !$acc end parallel

! CHECK: acc.parallel num_gangs({%c2{{.*}} : i32}, {%c1{{.*}} : i32, %c1{{.*}} : i32, %c1{{.*}} : i32} [#acc.device_type<nvidia>])

  !$acc parallel device_type(nvidia, default) num_gangs(1, 1, 1)
  !$acc end parallel

! CHECK: acc.parallel num_gangs({%c1{{.*}} : i32, %c1{{.*}} : i32, %c1{{.*}} : i32} [#acc.device_type<nvidia>], {%c1{{.*}} : i32, %c1{{.*}} : i32, %c1{{.*}} : i32} [#acc.device_type<default>])

end subroutine
