! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
module m1
  interface
    module subroutine s()
    end subroutine
  end interface
end

module m2
  interface
    module subroutine s()
    end subroutine
  end interface
end

submodule(m1) s1
end

!ERROR: Cannot read module file for submodule 's1' of module 'm2': Source file 'm2-s1.mod' was not found
submodule(m2:s1) s2
end

!ERROR: Cannot read module file for module 'm3': Source file 'm3.mod' was not found
submodule(m3:s1) s3
end
