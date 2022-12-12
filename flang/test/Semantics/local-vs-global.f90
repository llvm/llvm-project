! RUN: %python %S/test_errors.py %s %flang_fc1

module module_before_1
end

module module_before_2
end

block data block_data_before_1
end

block data block_data_before_2
end

subroutine explicit_before_1(a)
  real, optional :: a
end

subroutine explicit_before_2(a)
  real, optional :: a
end

subroutine implicit_before_1(a)
  real :: a
end

subroutine implicit_before_2(a)
  real :: a
end

function explicit_func_before_1(a)
  real, optional :: a
end

function explicit_func_before_2(a)
  real, optional :: a
end

function implicit_func_before_1(a)
  real :: a
end

function implicit_func_before_2(a)
  real :: a
end

program test
  external justfine ! OK to name a BLOCK DATA if not called
  !ERROR: The global entity 'module_before_1' corresponding to the local procedure 'module_before_1' is not a callable subprogram
  external module_before_1
  !ERROR: The global entity 'block_data_before_1' corresponding to the local procedure 'block_data_before_1' is not a callable subprogram
  external block_data_before_1
  !ERROR: The global subprogram 'explicit_before_1' may not be referenced via the implicit interface 'explicit_before_1'
  external explicit_before_1
  external implicit_before_1
  !ERROR: The global subprogram 'explicit_func_before_1' may not be referenced via the implicit interface 'explicit_func_before_1'
  external explicit_func_before_1
  external implicit_func_before_1
  !ERROR: The global entity 'module_after_1' corresponding to the local procedure 'module_after_1' is not a callable subprogram
  external module_after_1
  !ERROR: The global entity 'block_data_after_1' corresponding to the local procedure 'block_data_after_1' is not a callable subprogram
  external block_data_after_1
  !ERROR: The global subprogram 'explicit_after_1' may not be referenced via the implicit interface 'explicit_after_1'
  external explicit_after_1
  external implicit_after_1
  !ERROR: The global subprogram 'explicit_func_after_1' may not be referenced via the implicit interface 'explicit_func_after_1'
  external explicit_func_after_1
  external implicit_func_after_1
  call module_before_1
  !ERROR: 'module_before_2' is not a callable procedure
  call module_before_2
  call block_data_before_1
  !ERROR: 'block_data_before_2' is not a callable procedure
  call block_data_before_2
  call explicit_before_1(1.)
  !ERROR: References to the procedure 'explicit_before_2' require an explicit interface
  call explicit_before_2(1.)
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  call implicit_before_1
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  call implicit_before_2
  print *, explicit_func_before_1(1.)
  !ERROR: References to the procedure 'explicit_func_before_2' require an explicit interface
  print *, explicit_func_before_2(1.)
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  print *, implicit_func_before_1()
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  print *, implicit_func_before_2()
  call module_after_1
  call module_after_2
  call block_data_after_1
  call block_data_after_2
  call explicit_after_1(1.)
  !ERROR: References to the procedure 'explicit_after_2' require an explicit interface
  call explicit_after_2(1.)
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  call implicit_after_1
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  call implicit_after_2
  print *, explicit_func_after_1(1.)
  !ERROR: References to the procedure 'explicit_func_after_2' require an explicit interface
  print *, explicit_func_after_2(1.)
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  print *, implicit_func_after_1()
  !WARNING: If the procedure's interface were explicit, this reference would be in error
  !BECAUSE: Dummy argument 'a=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  print *, implicit_func_after_2()
end program

block data justfine
end

module module_after_1
end

!ERROR: 'module_after_2' is already declared in this scoping unit
module module_after_2
end

block data block_data_after_1
end

!ERROR: BLOCK DATA 'block_data_after_2' has been called
block data block_data_after_2
end

subroutine explicit_after_1(a)
  real, optional :: a
end

subroutine explicit_after_2(a)
  real, optional :: a
end

subroutine implicit_after_1(a)
  real :: a
end

subroutine implicit_after_2(a)
  real :: a
end

function explicit_func_after_1(a)
  real, optional :: a
end

function explicit_func_after_2(a)
  real, optional :: a
end

function implicit_func_after_1(a)
  real :: a
end

function implicit_func_after_2(a)
  real :: a
end
