! Test lowering of elemental character MIN/MAX to HLFIR
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test elemental character MIN with two array arguments of the same length.
subroutine test_elemental_char_min(a, b, res)
  character(5) :: a(10), b(10), res(10)
  res = min(a, b)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_min(
! CHECK:           %[[C5_A:.*]] = arith.constant 5 : index
! CHECK:           %[[A:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:           %[[C5_B:.*]] = arith.constant 5 : index
! CHECK:           %[[B:.*]]:2 = hlfir.declare {{.*}}Eb"
! CHECK:           %[[RES:.*]]:2 = hlfir.declare {{.*}}Eres"
! CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[RESULT_LEN:.*]] = arith.select %[[CMP]], %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %[[RESULT_LEN]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[IDX:.*]]: index):
! CHECK:             %{{.*}} = hlfir.char_extremum min, %{{.*}}, %{{.*}} :
! CHECK:             hlfir.yield_element
! CHECK:           }

! Test elemental character MAX with two array arguments.
subroutine test_elemental_char_max(a, b, res)
  character(5) :: a(10), b(10), res(10)
  res = max(a, b)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_max(
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %{{.*}} unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[IDX:.*]]: index):
! CHECK:             %{{.*}} = hlfir.char_extremum max, %{{.*}}, %{{.*}} :
! CHECK:             hlfir.yield_element
! CHECK:           }

! Test elemental character MIN with different argument lengths.
! The result length must be the maximum of the argument lengths.
subroutine test_elemental_char_min_diff_len(a, b, res)
  character(5) :: a(10)
  character(7) :: b(10), res(10)
  res = min(a, b)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_min_diff_len(
! CHECK:           %[[C5:.*]] = arith.constant 5 : index
! CHECK:           %[[A:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:           %[[C7:.*]] = arith.constant 7 : index
! CHECK:           %[[B:.*]]:2 = hlfir.declare {{.*}}Eb"
! CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[C7]], %[[C5]] : index
! CHECK:           %[[RESULT_LEN:.*]] = arith.select %[[CMP]], %[[C7]], %[[C5]] : index
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %[[RESULT_LEN]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,?>> {

! Test elemental character MIN with three array arguments.
! The result length must be the maximum across all argument lengths.
subroutine test_elemental_char_min_three(a, b, c, res)
  character(5) :: a(10), b(10), c(10), res(10)
  res = min(a, b, c)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_min_three(
! CHECK:           %[[C5_A:.*]] = arith.constant 5 : index
! CHECK:           %[[A:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:           %[[C5_B:.*]] = arith.constant 5 : index
! CHECK:           %[[B:.*]]:2 = hlfir.declare {{.*}}Eb"
! CHECK:           %[[C5_C:.*]] = arith.constant 5 : index
! CHECK:           %[[C:.*]]:2 = hlfir.declare {{.*}}Ec"
! CHECK:           %[[CMP1:.*]] = arith.cmpi sgt, %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[MAX1:.*]] = arith.select %[[CMP1]], %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[CMP2:.*]] = arith.cmpi sgt, %[[C5_C]], %[[MAX1]] : index
! CHECK:           %[[RESULT_LEN:.*]] = arith.select %[[CMP2]], %[[C5_C]], %[[MAX1]] : index
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %[[RESULT_LEN]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,?>> {
! CHECK:           ^bb0(%[[IDX:.*]]: index):
! CHECK:             %{{.*}} = hlfir.char_extremum min, %{{.*}}, %{{.*}}, %{{.*}} :
! CHECK:             hlfir.yield_element
! CHECK:           }

! Test elemental character MIN with an optional dummy argument of fixed length.
! The result length must guard the optional argument's length with fir.if so
! genCharLength is not called on an absent optional (unsafe for assumed length).
subroutine test_elemental_char_min_optional(a, b, c, res)
  character(5) :: a(10), b(10), res(10)
  character(5), optional :: c(10)
  res = min(a, b, c)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_min_optional(
! CHECK:           %[[C5_A:.*]] = arith.constant 5 : index
! CHECK:           %[[A:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:           %[[C5_B:.*]] = arith.constant 5 : index
! CHECK:           %[[B:.*]]:2 = hlfir.declare {{.*}}Eb"
! CHECK:           %[[C5_C:.*]] = arith.constant 5 : index
! CHECK:           %[[C:.*]]:2 = hlfir.declare {{.*}}Ec"
! CHECK:           %[[IS_PRESENT:.*]] = fir.is_present %[[C]]#0
! CHECK:           %[[CMP1:.*]] = arith.cmpi sgt, %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[MAX1:.*]] = arith.select %[[CMP1]], %[[C5_B]], %[[C5_A]] : index
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[OPT_LEN:.*]] = fir.if %[[IS_PRESENT]] -> (index) {
! CHECK:             fir.result %[[C5_C]] : index
! CHECK:           } else {
! CHECK:             fir.result %[[C0]] : index
! CHECK:           }
! CHECK:           %[[CMP2:.*]] = arith.cmpi sgt, %[[OPT_LEN]], %[[MAX1]] : index
! CHECK:           %[[RESULT_LEN:.*]] = arith.select %[[CMP2]], %[[OPT_LEN]], %[[MAX1]] : index
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %[[RESULT_LEN]]

! Test elemental character MIN with an assumed-length optional dummy argument.
! genCharLength reads from the descriptor, so it must be guarded with fir.if
! to avoid reading an invalid descriptor when the argument is absent.
subroutine test_elemental_char_min_assumed_optional(a, b, c, res)
  character(*), intent(in) :: a(:), b(:)
  character(*), intent(in), optional :: c(:)
  character(*), intent(out) :: res(:)
  res = min(a, b, c)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_elemental_char_min_assumed_optional(
! CHECK:           %[[A:.*]]:2 = hlfir.declare {{.*}}Ea"
! CHECK:           %[[B:.*]]:2 = hlfir.declare {{.*}}Eb"
! CHECK:           %[[C:.*]]:2 = hlfir.declare {{.*}}Ec"
! CHECK:           %[[IS_PRESENT:.*]] = fir.is_present %[[C]]#0
! CHECK:           %[[LEN_A:.*]] = fir.box_elesize %[[A]]#1
! CHECK:           %[[LEN_B:.*]] = fir.box_elesize %[[B]]#1
! CHECK:           %[[CMP1:.*]] = arith.cmpi sgt, %[[LEN_B]], %[[LEN_A]] : index
! CHECK:           %[[MAX1:.*]] = arith.select %[[CMP1]], %[[LEN_B]], %[[LEN_A]] : index
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[OPT_LEN:.*]] = fir.if %[[IS_PRESENT]] -> (index) {
! CHECK:             %[[LEN_C:.*]] = fir.box_elesize %[[C]]#1
! CHECK:             fir.result %[[LEN_C]] : index
! CHECK:           } else {
! CHECK:             fir.result %[[C0]] : index
! CHECK:           }
! CHECK:           %[[CMP2:.*]] = arith.cmpi sgt, %[[OPT_LEN]], %[[MAX1]] : index
! CHECK:           %[[RESULT_LEN:.*]] = arith.select %[[CMP2]], %[[OPT_LEN]], %[[MAX1]] : index
! CHECK:           %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} typeparams %[[RESULT_LEN]] unordered
