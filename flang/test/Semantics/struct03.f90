! RUN: %python %S/test_errors.py %s %flang_fc1
  structure /s/
    !ERROR: entity declarations are required on a nested structure
    structure /nested/
    end structure
  end structure
end
