! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of DOT_PRODUCT()
module m
  logical, parameter :: test_i4a = dot_product([(j,j=1,10)],[(j,j=1,10)]) == sum([(j*j,j=1,10)])
  logical, parameter :: test_r4a = dot_product([(1.*j,j=1,10)],[(j,j=1,10)]) == sum([(j*j,j=1,10)])
  logical, parameter :: test_z4a = dot_product([((j,j),j=1,10)],[((j,j),j=1,10)]) == sum([(((j,-j)*(j,j)),j=1,10)])
  logical, parameter :: test_l4a = .not. dot_product([logical::],[logical::])
  logical, parameter :: test_l4b = .not. dot_product([(j==2,j=1,10)], [(j==3,j=1,10)])
  logical, parameter :: test_l4c = dot_product([(j==4,j=1,10)], [(j==4,j=1,10)])
end
