! This test checks lowering of OpenACC data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

subroutine acc_data
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d, e
  logical :: ifCondition = .TRUE.

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! HLFIR: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! HLFIR: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! HLFIR: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]
! CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
! HLFIR: %[[DECLD:.*]]:2 = hlfir.declare %[[D]]
! CHECK: %[[E:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "e", uniq_name = "{{.*}}Ee"}
! HLFIR: %[[DECLE:.*]]:2 = hlfir.declare %[[E]]

  !$acc data if(.TRUE.) copy(a)
  !$acc end data

! CHECK:      %[[IF1:.*]] = arith.constant true
! FIR:        %[[COPYIN:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      acc.data if(%[[IF1]]) dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>)  {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:   acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}

  !$acc data copy(a) if(ifCondition)
  !$acc end data

! FIR:        %[[COPYIN:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK:      %[[IFCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK:      %[[IF2:.*]] = fir.convert %[[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK:      acc.data if(%[[IF2]]) dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:   acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}

  !$acc data copy(a, b, c)
  !$acc end data

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! FIR:        %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! HLFIR:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:   acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:   acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! FIR:   acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}

  !$acc data copy(a) copy(b) copy(c)
  !$acc end data

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
! FIR:        %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! HLFIR:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "c"}
! CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:   acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! FIR:   acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "b"}
! FIR:   acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}
! HLFIR: acc.copyout accPtr(%[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "c"}

  !$acc data copyin(a) copyin(readonly: b, c)
  !$acc end data

! FIR:        %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! HLFIR:      %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! FIR:        %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! HLFIR:      %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! FIR:        %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "c"}
! HLFIR:      %[[COPYIN_C:.*]] = acc.copyin varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "c"}
! CHECK:      acc.data dataOperands(%[[COPYIN_A]], %[[COPYIN_B]], %[[COPYIN_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc data copyout(a) copyout(zero: b) copyout(c)
  !$acc end data

! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout_zero>, name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout_zero>, name = "b"}
! FIR:        %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "c"}
! HLFIR:      %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "c"}
! CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! FIR:   acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a"}
! HLFIR: acc.copyout accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a"}
! FIR:   acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copyout_zero>, name = "b"}
! HLFIR: acc.copyout accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_copyout_zero>, name = "b"}
! FIR:   acc.copyout accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {name = "c"}
! HLFIR: acc.copyout accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "c"}

  !$acc data create(a, b) create(zero: c)
  !$acc end data

! FIR:        %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! HLFIR:      %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! FIR:        %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! HLFIR:      %[[CREATE_B:.*]] = acc.create varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! FIR:        %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! HLFIR:      %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! CHECK:      acc.data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK: acc.delete accPtr(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "a"}
! CHECK: acc.delete accPtr(%[[CREATE_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create>, name = "b"}
! CHECK: acc.delete accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "c"}

  !$acc data create(c) copy(b) create(a)
  !$acc end data
!FIR:   %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
!HLFIR: %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
!FIR:   %[[COPY_B:.*]] = acc.copyin varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
!HLFIR: %[[COPY_B:.*]] = acc.copyin varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copy>, name = "b"}
!FIR:   %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
!HLFIR: %[[CREATE_A:.*]] = acc.create varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
!CHECK: acc.data dataOperands(%[[CREATE_C]], %[[COPY_B]], %[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {

  !$acc data no_create(a, b) create(zero: c)
  !$acc end data

! FIR:        %[[NO_CREATE_A:.*]] = acc.nocreate varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! HLFIR:      %[[NO_CREATE_A:.*]] = acc.nocreate varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! FIR:        %[[NO_CREATE_B:.*]] = acc.nocreate varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! HLFIR:      %[[NO_CREATE_B:.*]] = acc.nocreate varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! FIR:        %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! HLFIR:      %[[CREATE_C:.*]] = acc.create varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_create_zero>, name = "c"}
! CHECK:      acc.data dataOperands(%[[NO_CREATE_A]], %[[NO_CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK: acc.delete accPtr(%[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) {dataClause = #acc<data_clause acc_create_zero>, name = "c"}

  !$acc data present(a, b, c)
  !$acc end data

! FIR:        %[[PRESENT_A:.*]] = acc.present varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! HLFIR:      %[[PRESENT_A:.*]] = acc.present varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "a"}
! FIR:        %[[PRESENT_B:.*]] = acc.present varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! HLFIR:      %[[PRESENT_B:.*]] = acc.present varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! FIR:        %[[PRESENT_C:.*]] = acc.present varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! HLFIR:      %[[PRESENT_C:.*]] = acc.present varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.data dataOperands(%[[PRESENT_A]], %[[PRESENT_B]], %[[PRESENT_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc data deviceptr(b, c)
  !$acc end data

! FIR:        %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! HLFIR:      %[[DEVICEPTR_B:.*]] = acc.deviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "b"}
! FIR:        %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! HLFIR:      %[[DEVICEPTR_C:.*]] = acc.deviceptr varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c"}
! CHECK:      acc.data dataOperands(%[[DEVICEPTR_B]], %[[DEVICEPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}

  !$acc data attach(d, e)
  !$acc end data

! CHECK:      %[[ATTACH_D:.*]] = acc.attach varPtr(%{{.*}} : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "d"}
! CHECK:      %[[ATTACH_E:.*]] = acc.attach varPtr(%{{.*}} : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "e"}
! CHECK:      acc.data dataOperands(%[[ATTACH_D]], %[[ATTACH_E]] : !fir.ptr<f32>, !fir.ptr<f32>) {
! CHECK:        acc.terminator
! CHECK-NEXT: }{{$}}
! CHECK: acc.detach accPtr(%[[ATTACH_D]] : !fir.ptr<f32>) {dataClause = #acc<data_clause acc_attach>, name = "d"}
! CHECK: acc.detach accPtr(%[[ATTACH_E]] : !fir.ptr<f32>) {dataClause = #acc<data_clause acc_attach>, name = "e"}

  !$acc data present(a) async
  !$acc end data

! CHECK: acc.data dataOperands(%{{.*}}) {
! CHECK: } attributes {asyncAttr}

  !$acc data present(a) async(1)
  !$acc end data

! CHECK: acc.data async(%{{.*}} : i32) dataOperands(%{{.*}}) {
! CHECK: }{{$}}

  !$acc data present(a) wait
  !$acc end data

! CHECK: acc.data dataOperands(%{{.*}}) {
! CHECK: } attributes {waitAttr}

  !$acc data present(a) wait(1)
  !$acc end data

! CHECK: acc.data dataOperands(%{{.*}}) wait(%{{.*}} : i32) {
! CHECK: }{{$}}

  !$acc data present(a) wait(devnum: 0: 1)
  !$acc end data

! CHECK: acc.data dataOperands(%{{.*}}) wait_devnum(%{{.*}} : i32) wait(%{{.*}} : i32) {
! CHECK: }{{$}}

  !$acc data default(none)
  !$acc end data

! CHECK: acc.data {
! CHECK:   acc.terminator
! CHECK: } attributes {defaultAttr = #acc<defaultvalue none>}

  !$acc data default(present)
  !$acc end data

! CHECK: acc.data {
! CHECK:   acc.terminator
! CHECK: } attributes {defaultAttr = #acc<defaultvalue present>}

  !$acc data
  !$acc end data
! CHECK-NOT: acc.data

end subroutine acc_data

