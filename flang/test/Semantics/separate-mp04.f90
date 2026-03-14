! RUN: %python %S/test_errors.py %s %flang_fc1
! Checks for multiple module procedure definitions

module m1
  interface
    module subroutine x001
    end subroutine
    module subroutine x002
    end subroutine
    module subroutine x003
    end subroutine
  end interface
end

submodule(m1) sm1
  interface
    module subroutine x004
    end subroutine
  end interface
 contains
  module procedure x001 ! fine
  end procedure
  module subroutine x002
  end subroutine
  module subroutine x003
  end subroutine
end

submodule(m1) sm2
 contains
  !ERROR: Module procedure 'x002' in 'm1' has multiple definitions
  module subroutine x002
  end subroutine
end

submodule(m1:sm2) sm3
 contains
  !ERROR: Module procedure 'x002' in 'm1' has multiple definitions
  module subroutine x002
  end subroutine
  !ERROR: Module procedure 'x003' in 'm1' has multiple definitions
  module subroutine x003
  end subroutine
end

submodule(m1:sm1) sm4
 contains
  module subroutine x004
  end subroutine
end

submodule(m1:sm1) sm5
 contains
  !ERROR: Module procedure 'x004' in 'm1:sm1' has multiple definitions
  module subroutine x004
  end subroutine
end
