!RUN: %flang -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!Ensure no bogus error messages about insufficiently defined procedures
!CHECK-NOT: error

module m
  interface
    module subroutine smp1(a1)
    end
  end interface
end

submodule(m) sm1
  interface
    module subroutine smp2(a1,a2)
    end
  end interface
end

submodule(m:sm1) sm2
  interface generic
    procedure smp1
    procedure smp2
    module subroutine smp3(a1,a2,a3)
    end
  end interface
 contains
  subroutine local1
    call generic(0.)
    call generic(0., 1.)
    call generic(0., 1., 2.)
  end
  subroutine local2(a1,a2,a3)
  end
  module procedure smp1
  end
  module subroutine smp2(a1,a2)
  end
  module subroutine smp3(a1,a2,a3)
  end
end


