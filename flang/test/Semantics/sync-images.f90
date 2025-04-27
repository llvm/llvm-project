!RUN: %python %S/test_errors.py %s %flang_fc1
integer twod(1,1)
sync images (*) ! ok
!ERROR: An image-set that is an int-expr must be a scalar or a rank-one array
sync images (twod)
!ERROR: Must have INTEGER type, but is REAL(4)
sync images (3.14159)
!ERROR: Image number -1 in the image-set is not valid
sync images (-1)
!ERROR: Image number -1 in the image-set is not valid
sync images ([2, -1, 3])
end
