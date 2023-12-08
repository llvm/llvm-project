! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of TRANSFER(...)

module m
  logical, parameter :: test_r2i_s_1 = transfer(1., 0) == int(z'3f800000')
  logical, parameter :: test_r2i_v_1 = all(transfer(1., [integer::]) == [int(z'3f800000')])
  logical, parameter :: test_r2i_v_2 = all(transfer([1., 2.], [integer::]) == [int(z'3f800000'), int(z'40000000')])
  logical, parameter :: test_r2i_vs_1 = all(transfer([1., 2.], [integer::], 1) == [int(z'3f800000')])

  type :: t
    real :: x = 0.
  end type t
  logical, parameter :: test_t2i_s_1 = transfer(t(1.), 0) == int(z'3f800000')
  logical, parameter :: test_t2i_v_1 = all(transfer(t(1.), [integer::]) == [int(z'3f800000')])
  logical, parameter :: test_t2i_v_2 = all(transfer([t(1.), t(2.)], [integer::]) == [int(z'3f800000'), int(z'40000000')])
  logical, parameter :: test_t2i_vs_1 = all(transfer([t(1.), t(2.)], [integer::], 1) == [int(z'3f800000')])

  type(t), parameter :: t1 = transfer(1., t())
  logical, parameter :: test_r2t_s_1 = t1%x == 1.
  type(t), parameter :: t2(*) = transfer(1., [t::])
  logical, parameter :: test_r2t_v_1 = all(t2%x == [1.])
  type(t), parameter :: t3(*) = transfer([1., 2.], [t::])
  logical, parameter :: test_r2t_v_2 = all(t3%x == [1., 2.])
  type(t), parameter :: t4(*) = transfer([1., 2.], t(), 1)
  logical, parameter :: test_r2t_vs_1 = all(t4%x == [1.])

  logical, parameter :: test_nan = transfer(int(z'7ff8000000000000', 8), 0._8) /= transfer(int(z'7ff8000000000000', 8), 0._8)

  integer, parameter :: jc1 = transfer("abcd", 0)
  logical, parameter :: test_c2i_s_1 = jc1 == int(z'61626364') .or. jc1 == int(z'64636261')
  integer, parameter :: jc2(*) = transfer("abcd", [integer::])
  logical, parameter :: test_c2i_v_1 = all(jc2 == int(z'61626364') .or. jc1 == int(z'64636261'))
  integer, parameter :: jc3(*) = transfer(["abcd", "efgh"], [integer::])
  logical, parameter :: test_c2i_v_2 = all(jc3 == [int(z'61626364'), int(z'65666768')]) .or. all(jc3 == [int(z'64636261'), int(z'68676665')])
  integer, parameter :: jc4(*) = transfer(["abcd", "efgh"], 0, 1)
  logical, parameter :: test_c2i_vs_1 = all(jc4 == [int(z'61626364')]) .or. all(jc4 == [int(z'64636261')])

  integer, parameter :: le1 = int(z'64636261', 4), be1 = int(z'65666768', 4)
  character*5, parameter :: le1c(*) = transfer(le1, [character(5)::])
  character*5, parameter :: be1c(*) = transfer(be1, [character(5)::])
  logical, parameter :: test_i2c_s = all(le1c == ["abcd"//char(0)]) .or. all(be1c == ["efgh"//char(0)])

  character*4, parameter :: i2ss1 = transfer(int(z'61626364', 4), "12345678"(2:5))
  logical, parameter :: test_i2ss1 = any(i2ss1 == ["abcd", "dcba"])
end module
