! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  interface
    module subroutine separate
    end
  end interface
 contains
  subroutine modsub
    !ERROR: Module 'm' cannot USE itself
    use m
  end
end

submodule(m) submod1
 contains
  module subroutine separate
    !ERROR: Module 'm' cannot USE itself from its own submodule 'submod1'
    use m
  end
end

submodule(m) submod2
  !ERROR: Module 'm' cannot USE itself from its own submodule 'submod2'
  use m
end

