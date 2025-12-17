! RUN: %flang %s -o %t && %t | FileCheck %s
! Test F_C_STRING library function

program test_f_c_string
  use iso_c_binding
  implicit none
  
  character(len=20) :: str
  character(len=:), allocatable :: result
  logical :: flag
  
  ! Test 1: Basic trimming
  str = 'hello     '
  result = f_c_string(str(1:10))
  ! CHECK: Test 1: 6
  print '(A,I0)', 'Test 1: ', len(result)
  if (result /= 'hello' // c_null_char) error stop 'Test 1 failed'
  
  ! Test 2: ASIS=.TRUE. (keep blanks)
  result = f_c_string(str(1:10), .true.)
  ! CHECK: Test 2: 11
  print '(A,I0)', 'Test 2: ', len(result)
  if (result /= 'hello     ' // c_null_char) error stop 'Test 2 failed'
  
  ! Test 3: ASIS=.FALSE. (explicit trim)
  result = f_c_string(str(1:10), .false.)
  ! CHECK: Test 3: 6
  print '(A,I0)', 'Test 3: ', len(result)
  if (result /= 'hello' // c_null_char) error stop 'Test 3 failed'
  
  ! Test 4: Variable ASIS
  flag = .true.
  str = 'abc   '
  result = f_c_string(str(1:6), flag)
  ! CHECK: Test 4: 7
  print '(A,I0)', 'Test 4: ', len(result)
  if (len(result) /= 7) error stop 'Test 4 failed'
  
  flag = .false.
  result = f_c_string(str(1:6), flag)
  ! CHECK: Test 5: 4
  print '(A,I0)', 'Test 5: ', len(result)
  if (len(result) /= 4) error stop 'Test 5 failed'
  
  ! Test 6: Empty (all blanks)
  result = f_c_string('     ')
  ! CHECK: Test 6: 1
  print '(A,I0)', 'Test 6: ', len(result)
  if (result /= c_null_char) error stop 'Test 6 failed'
  
  ! Test 7: Internal blanks preserved
  result = f_c_string('a b c   ')
  ! CHECK: Test 7: 6
  print '(A,I0)', 'Test 7: ', len(result)
  if (result /= 'a b c' // c_null_char) error stop 'Test 7 failed'
  
  ! CHECK: PASS
  print *, 'PASS'
  
end program test_f_c_string
