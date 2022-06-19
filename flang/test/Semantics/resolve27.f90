! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  interface
    module subroutine s()
    end subroutine
  end interface
end

submodule(m) s1
end

submodule(m) s2
end

submodule(m:s1) s3
  integer x
end

!ERROR: Module 'm' already has a submodule named 's3'
submodule(m:s2) s3
  integer y
end
