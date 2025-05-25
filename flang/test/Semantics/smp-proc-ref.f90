!RUN: %flang_fc1 -fsyntax-only %s
module m
  real :: qux(10)
  interface
    module subroutine bar(i)
    end
    module function baz()
    end
  end interface
end

submodule(m) sm
 contains
  module procedure bar
    qux(i) = baz() ! ensure no bogus error here
  end
  module procedure baz
    baz = 1.
  end
end
