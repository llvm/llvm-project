! RUN: %python %S/test_errors.py %s %flang_fc1
! C815: an attribute may be applied at most once per scope
module m
  real a1, a2, v1, v2
  asynchronous a1
  asynchronous a2
  !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
  asynchronous a2
  volatile v1
  volatile v2
  !ERROR: VOLATILE attribute was already specified on 'v2'
  volatile v2
 contains
  subroutine modsub
    asynchronous a1
    asynchronous a2
    !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
    asynchronous a2
    volatile v1
    volatile v2
    !ERROR: VOLATILE attribute was already specified on 'v2'
    volatile v2
    block
      asynchronous a1
      asynchronous a2
      !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
      asynchronous a2
      volatile v1
      volatile v2
      !ERROR: VOLATILE attribute was already specified on 'v2'
      volatile v2
    end block
  end
end

subroutine s
  use m
  asynchronous a1
  asynchronous a2
  !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
  asynchronous a2
  volatile v1
  volatile v2
  !ERROR: VOLATILE attribute was already specified on 'v2'
  volatile v2
  block
    asynchronous a1
    asynchronous a2
    !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
    asynchronous a2
    volatile v1
    volatile v2
    !ERROR: VOLATILE attribute was already specified on 'v2'
    volatile v2
  end block
 contains
  subroutine internal
    asynchronous a1
    asynchronous a2
    !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
    asynchronous a2
    volatile v1
    volatile v2
    !ERROR: VOLATILE attribute was already specified on 'v2'
    volatile v2
    block
      asynchronous a1
      asynchronous a2
      !ERROR: ASYNCHRONOUS attribute was already specified on 'a2'
      asynchronous a2
      volatile v1
      volatile v2
      !ERROR: VOLATILE attribute was already specified on 'v2'
      volatile v2
    end block
  end
end

