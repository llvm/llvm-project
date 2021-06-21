! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests intrinsic MAXVAL/MINVAL function folding
module m
  logical, parameter :: test_imaxidentity = maxval([integer::]) == -huge(0) - 1
  logical, parameter :: test_iminidentity = minval([integer::]) == huge(0)
  integer, parameter :: intmatrix(*,*) = reshape([1, 2, 3, 4, 5, 6], [2, 3])
  logical, parameter :: test_imaxval = maxval(intmatrix) == 6
  logical, parameter :: test_iminval = minval(intmatrix) == 1
  logical, parameter :: odds(2,3) = mod(intmatrix, 2) == 1
  logical, parameter :: test_imaxval_masked = maxval(intmatrix,odds) == 5
  logical, parameter :: test_iminval_masked = minval(intmatrix,.not.odds) == 2
  logical, parameter :: test_rmaxidentity = maxval([real::]) == -huge(0.0)
  logical, parameter :: test_rminidentity = minval([real::]) == huge(0.0)
  logical, parameter :: test_rmaxval = maxval(real(intmatrix)) == 6.0
  logical, parameter :: test_rminval = minval(real(intmatrix)) == 1.0
  logical, parameter :: test_rmaxval_scalar_mask = maxval(real(intmatrix), .true.) == 6.0
  logical, parameter :: test_rminval_scalar_mask = minval(real(intmatrix), .false.) == huge(0.0)
  character(*), parameter :: chmatrix(*,*) = reshape(['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr'], [2, 3])
  logical, parameter :: test_cmaxlen = len(maxval([character*4::])) == 4
  logical, parameter :: test_cmaxidentity = maxval([character*4::]) == repeat(char(0), 4)
  logical, parameter :: test_cminidentity = minval([character*4::]) == repeat(char(127), 4)
  logical, parameter :: test_cmaxval = maxval(chmatrix) == 'pqr'
  logical, parameter :: test_cminval = minval(chmatrix) == 'abc'
  logical, parameter :: test_dim1 = all(maxval(intmatrix,dim=1) == [2, 4, 6])
  logical, parameter :: test_dim2 = all(minval(intmatrix,dim=2,mask=odds) == [1, huge(0)])
end
