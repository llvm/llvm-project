!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
implicit none

integer, parameter :: n=36
integer, dimension(n) :: rslts, expect

integer, dimension(9) :: some_array = (/ 5, 2, 3, 4, 1, 5, 1, 3, 4 /)
integer, parameter, dimension(9) :: some_parameter_array = (/ 5, 2, 3, 4, 1, 5, 1, 3, 4 /)
integer, dimension(1) :: loc

integer, dimension(1:3,1:3) :: some_matrix = reshape(some_parameter_array, (/ 3, 3 /))
integer, parameter, dimension(1:3,1:3) :: some_parameter_matrix = reshape(some_parameter_array, (/ 3, 3 /))

! array, constant input, constant output.
integer, parameter, dimension(1) :: &
        param_array_minloc = minloc(some_parameter_array), &
        param_array_minloc_back = minloc(some_parameter_array, back=.TRUE.), &
        param_array_maxloc = maxloc(some_parameter_array), &
        param_array_maxloc_back = maxloc(some_parameter_array, back=.TRUE.)

! matrix, constant input, constant output.
integer, parameter, dimension(2) :: &
        param_matrix_minloc = minloc(some_parameter_matrix), &
        param_matrix_minloc_back = minloc(some_parameter_matrix, back=.TRUE.), &
        param_matrix_maxloc = maxloc(some_parameter_matrix), &
        param_matrix_maxloc_back = maxloc(some_parameter_matrix, back=.TRUE.)

rslts(1:1) = minloc(some_array)
rslts(2:2) = minloc(some_array, back=.TRUE.)
rslts(3:3) = maxloc(some_array)
rslts(4:4) = maxloc(some_array, back=.TRUE.)

rslts(5:5) = minloc(some_parameter_array)
rslts(6:6) = minloc(some_parameter_array, back=.TRUE.)
rslts(7:7) = maxloc(some_parameter_array)
rslts(8:8) = maxloc(some_parameter_array, back=.TRUE.)

rslts(9:10) = minloc(some_matrix)
rslts(11:12) = minloc(some_matrix, back=.TRUE.)
rslts(13:14) = maxloc(some_matrix)
rslts(15:16) = maxloc(some_matrix, back=.TRUE.)

rslts(17:18) = minloc(some_parameter_matrix)
rslts(19:20) = minloc(some_parameter_matrix, back=.TRUE.)
rslts(21:22) = maxloc(some_parameter_matrix)
rslts(23:24) = maxloc(some_parameter_matrix, back=.TRUE.)

! rslts(25:25) = param_array_minloc
rslts(25:25) = (/ 5 /) ! test inhibited as param_array_minloc is not correct. This is issue flang-compiler/flang#763.
rslts(26:26) = param_array_minloc_back
rslts(27:27) = param_array_maxloc
rslts(28:28) = param_array_maxloc_back

! rslts(25:26) = param_matrix_minloc
rslts(29:30) = (/ 2, 2 /) ! test inhibited as param_matrix_minloc is not correct. This is issue flang-compiler/flang#763.
rslts(31:32) = param_matrix_minloc_back
rslts(33:34) = param_matrix_maxloc
rslts(35:36) = param_matrix_maxloc_back

call check(rslts, expect, n)

data expect / 5, 7, 1, 6, &
              5, 7, 1, 6, &

              ! some_matrix
              2, 2, &
              1, 3, &
              1, 1, &
              3, 2, &

              ! some_parameter_matrix
              2, 2, &
              1, 3, &
              1, 1, &
              3, 2, &

              ! param_array_{min,max}loc{,_back}
              5, 7, 1, 6, &

              ! param_matrix_{min,max}loc{,_back}
              2, 2, &
              1, 3, &
              1, 1, &
              3, 2 /

end
