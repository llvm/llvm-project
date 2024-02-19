! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of MERGE
module m
  type t
    integer n
  end type
  logical, parameter :: test_01 = all(merge([1,2,3],4,[.true.,.false.,.true.]) == [1,4,3])
  logical, parameter :: test_02 = all(merge([1,2,3],4,.true.) == [1,2,3])
  logical, parameter :: test_03 = all(merge([1,2,3],4,.false.) == [4,4,4])
  logical, parameter :: test_04 = all(merge(1,4,[.true.,.false.,.true.,.false.]) == [1,4,1,4])
  type(t), parameter :: dt00a = merge(t(1),t(2),.true.)
  logical, parameter :: test_05 = dt00a%n == 1
  type(t), parameter :: dt00b = merge(t(1),t(2),.false.)
  logical, parameter :: test_06 = dt00b%n == 2
  type(t), parameter :: dt01(*) = merge([t(1),t(2)],[t(3),t(4)],[.false.,.true.])
  logical, parameter :: test_07 = all(dt01%n == [3,2])
  type(t), parameter :: dt02(*) = merge(t(1),[t(3),t(4)],.true.)
  logical, parameter :: test_08 = all(dt02%n == [1,1])
  type(t), parameter :: dt03(*) = merge([t(1),t(2)],t(3),[.true.,.false.])
  logical, parameter :: test_09 = all(dt03%n == [1,3])
  logical, parameter :: test_10 = merge('ab','cd',.true.) == 'ab'
end
