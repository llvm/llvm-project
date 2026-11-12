! RUN: %python %S/test_errors.py %s %flang_fc1
interface a1
  subroutine s1
    interface a2
      subroutine s2
        !ERROR: Invalid specification expression: reference to local entity 'k'
        real x(k)
      end subroutine
    end interface
    !ERROR: Invalid specification expression: reference to local entity 'k'
    real y(k)
  end subroutine
end interface
end
