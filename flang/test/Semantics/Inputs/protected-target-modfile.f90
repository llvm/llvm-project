subroutine use_protected_target
  use protected_target_mod, only: renamed => p, holder, x, read_pointer, &
    ordinary_pointer, default_value, renamed_result => result_pointer
  type(holder) :: h
  integer, pointer :: q
  renamed => x
  h = holder(renamed)
  h%q => x
  call read_pointer(renamed)
  !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'a=' without PROTECTED_TARGET
  call ordinary_pointer(renamed)
  !ERROR: A PROTECTED_TARGET pointer or its subobject may be associated with nonpointer dummy argument 'a=' only with INTENT(IN)
  call default_value(h%q)
  !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
  q => renamed
  !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with pointer 'q' without PROTECTED_TARGET
  q => renamed_result()
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'renamed' is not definable
  renamed = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'renamed_result' is not definable
  renamed_result() = 1
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: The target of PROTECTED_TARGET pointer 'q' is not definable
  h%q = 1
  !ERROR: An ALLOCATE statement for a PROTECTED_TARGET pointer must have SOURCE=
  allocate(renamed)
  allocate(h%q, source=1)
  !ERROR: A PROTECTED_TARGET pointer may not be deallocated
  deallocate(h%q)
  nullify(renamed, h%q)
end subroutine
