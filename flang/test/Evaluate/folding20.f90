! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests reduction intrinsic function folding
module m
  implicit none
  integer, parameter :: intmatrix(*,*) = reshape([1, 2, 3, 4, 5, 6], [2, 3])
  logical, parameter :: odds(2,3) = mod(intmatrix, 2) == 1
  character(*), parameter :: chmatrix(*,*) = reshape(['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr'], [2, 3])
  real, parameter :: nan = real(z'7FC00000'), inf = real(z'7F800000')
  real, parameter :: nans(*) = [nan, nan]
  real, parameter :: someNan(*) = [nan, 0.]
  real, parameter :: someInf(*) = [inf, 0.]

  logical, parameter :: test_allidentity = all([Logical::])
  logical, parameter :: test_all = .not. all(odds)
  logical, parameter :: test_alldim1 = all(.not. all(odds,1))
  logical, parameter :: test_alldim2 = all(all(odds,2) .eqv. [.true., .false.])
  logical, parameter :: test_anyidentity = .not. any([Logical::])
  logical, parameter :: test_any = any(odds)
  logical, parameter :: test_anydim1 = all(any(odds,1))
  logical, parameter :: test_anydim2 = all(any(odds,2) .eqv. [.true., .false.])

  logical, parameter :: test_iallidentity = iall([integer::]) == -1
  logical, parameter :: test_iall = iall(intmatrix) == 0
  logical, parameter :: test_iall_masked = iall(intmatrix,odds) == 1
  logical, parameter :: test_ialldim1 = all(iall(intmatrix,dim=1) == [0, 0, 4])
  logical, parameter :: test_ialldim2 = all(iall(intmatrix,dim=2) == [1, 0])
  logical, parameter :: test_ianyidentity = iany([integer::]) == 0
  logical, parameter :: test_iany = iany(intmatrix) == 7
  logical, parameter :: test_iany_masked = iany(intmatrix,odds) == 7
  logical, parameter :: test_ianydim1 = all(iany(intmatrix,dim=1) == [3, 7, 7])
  logical, parameter :: test_ianydim2 = all(iany(intmatrix,dim=2) == [7, 6])
  logical, parameter :: test_iparityidentity = iparity([integer::]) == 0
  logical, parameter :: test_iparity = iparity(intmatrix) == 7
  logical, parameter :: test_iparity_masked = iparity(intmatrix,odds) == 7
  logical, parameter :: test_iparitydim1 = all(iparity(intmatrix,dim=1) == [3, 7, 3])
  logical, parameter :: test_iparitydim2 = all(iparity(intmatrix,dim=2) == [7, 0])

  logical, parameter :: test_imaxidentity = maxval([integer::]) == -huge(0) - 1
  logical, parameter :: test_iminidentity = minval([integer::]) == huge(0)
  logical, parameter :: test_imaxval = maxval(intmatrix) == 6
  logical, parameter :: test_iminval = minval(intmatrix) == 1
  logical, parameter :: test_imaxval_masked = maxval(intmatrix,odds) == 5
  logical, parameter :: test_iminval_masked = minval(intmatrix,.not.odds) == 2
  logical, parameter :: test_rmaxidentity = maxval([real::]) == -huge(0.0)
  logical, parameter :: test_rminidentity = minval([real::]) == huge(0.0)
  logical, parameter :: test_rmaxval = maxval(real(intmatrix)) == 6.0
  logical, parameter :: test_rminval = minval(real(intmatrix)) == 1.0
  logical, parameter :: test_rmaxval_scalar_mask = maxval(real(intmatrix), .true.) == 6.0
  logical, parameter :: test_rminval_scalar_mask = minval(real(intmatrix), .false.) == huge(0.0)
  logical, parameter :: test_rmaxval_allNaN = maxval(nans) /= maxval(nans)
  logical, parameter :: test_rminval_allNaN = maxval(nans) /= maxval(nans)
  logical, parameter :: test_rmaxval_someNaN = maxval(someNan) == 0.
  logical, parameter :: test_rminval_someNaN = minval(someNan) == 0.
  logical, parameter :: test_rmaxval_someInf = maxval(someInf) == inf
  logical, parameter :: test_rminval_someInf = minval(-someInf) == -inf
  logical, parameter :: test_cmaxlen = len(maxval([character*4::])) == 4
  logical, parameter :: test_cmaxidentity = maxval([character*4::]) == repeat(char(0), 4)
  logical, parameter :: test_cminidentity = minval([character*4::]) == repeat(char(255), 4)
  logical, parameter :: test_cmaxval = maxval(chmatrix) == 'pqr'
  logical, parameter :: test_cminval = minval(chmatrix) == 'abc'
  logical, parameter :: test_maxvaldim1 = all(maxval(intmatrix,dim=1) == [2, 4, 6])
  logical, parameter :: test_minvaldim2 = all(minval(intmatrix,dim=2,mask=odds) == [1, huge(0)])

  logical, parameter :: test_iproductidentity = product([integer::]) == 1
  logical, parameter :: test_iproduct = product(intmatrix) == 720
  logical, parameter :: test_iproduct_masked = product(intmatrix,odds) == 15
  logical, parameter :: test_productdim1 = all(product(intmatrix,dim=1) == [2, 12, 30])
  logical, parameter :: test_productdim2 = all(product(intmatrix,dim=2) == [15, 48])
  logical, parameter :: test_rproductidentity = product([real::]) == 1.
  logical, parameter :: test_rproduct = product(real(intmatrix)) == 720.
  logical, parameter :: test_cproductidentity = product([complex::]) == (1.,0.)
  logical, parameter :: test_cproduct = product(cmplx(intmatrix,-intmatrix)) == (0.,5760.)

  logical, parameter :: test_isumidentity = sum([integer::]) == 0
  logical, parameter :: test_isum = sum(intmatrix) == 21
  logical, parameter :: test_isum_masked = sum(intmatrix,odds) == 9
  logical, parameter :: test_sumdim1 = all(sum(intmatrix,dim=1) == [3, 7, 11])
  logical, parameter :: test_sumdim2 = all(sum(intmatrix,dim=2) == [9, 12])
  logical, parameter :: test_rsumidentity = sum([real::]) == 0.
  logical, parameter :: test_rsum = sum(real(intmatrix)) == 21.
  logical, parameter :: test_csumidentity = sum([complex::]) == (0.,0.)
  logical, parameter :: test_csum = sum(cmplx(intmatrix,-intmatrix)) == (21.,-21.)
end
