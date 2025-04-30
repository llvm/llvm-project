!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program mmul
  integer, parameter :: n = 4
  real(kind = 8), dimension(1:n, 1:n) :: a, b, c
  real(kind = 8), dimension(n * n) :: expected
  data expected / -256.0, -256.0, -256.0, -256.0, &
                  -256.0, -256.0, -256.0, -256.0, &
                  -256.0, -256.0, -256.0, -256.0, &
                  -256.0, -256.0, -256.0, -256.0 /

  a = 1.0
  b = -1.0
  c = 2.0
  call do_matmul(a, b, c, n)
  call checkd(reshape(c, (/ n * n /)), expected, n * n)

contains

  subroutine do_matmul(a, b, c, n)
    integer, intent(in) :: n
    real(kind = 8), dimension(:, :), intent(in) :: a, b
    real(kind = 8), dimension(:, :), intent(inout) :: c

    c(1:n, 1:n) = matmul(matmul(matmul(a(1:n, 1:n), b(1:n, 1:n)), matmul(a(1:n, 1:n), b(1:n, 1:n))), b(1:n, 1:n))
  end subroutine

end program mmul
