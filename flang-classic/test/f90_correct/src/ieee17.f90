!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee17
use ieee_arithmetic
real*8 a, sqa, sqa0, sqa1, sqa2, sqa3, sqa4
type(ieee_round_type) :: ra, ra0, ra1, ra2, ra3
type(ieee_round_type) :: sa, sa0, sa1, sa2, sa3
logical lres(15), lexp(15)

a = 5.0d0
call ieee_get_rounding_mode(ra)
sqa = sqrt(a * ieee_value(a,ieee_positive_normal))

ra0 = ieee_nearest
call ieee_set_rounding_mode(ra0)
sqa0 = sqrt(a * ieee_value(a,ieee_positive_normal))
call ieee_get_rounding_mode(sa0)
print *,(sqa .eq. sqa0),(ra0 .eq. sa0)
lres(1) = (sqa .eq. sqa0)
lres(2) = (ra0 .eq. sa0)
lres(3) = (ieee_nearest .eq. sa0)

ra1 = ieee_to_zero
call ieee_set_rounding_mode(ra1)
sqa1 = sqrt(a * ieee_value(a,ieee_positive_normal))
call ieee_get_rounding_mode(sa1)
print *,(sqa .eq. sqa1),(ra1 .eq. sa1),(sqa1 .eq. ieee_next_after(sqa, 0.0d0))
lres(4) = (sqa .eq. sqa1) .or. (sqa1 .eq. ieee_next_after(sqa, 0.0d0))
lres(5) = (ra1 .eq. sa1)
lres(6) = (ieee_to_zero .eq. sa1)

ra2 = ieee_up
call ieee_set_rounding_mode(ra2)
sqa2 = sqrt(a * ieee_value(a,ieee_positive_normal))
call ieee_get_rounding_mode(sa2)
print *,(sqa .eq. sqa2),(ra2 .eq. sa2),(sqa2 .eq. ieee_next_after(sqa, a))
lres(7) = (sqa .eq. sqa2) .or. (sqa2 .eq. ieee_next_after(sqa, a))
lres(8) = (ra2 .eq. sa2)
lres(9) = (ieee_up .eq. sa2)

ra3 = ieee_down
call ieee_set_rounding_mode(ra3)
sqa3 = sqrt(a * ieee_value(a,ieee_positive_normal))
call ieee_get_rounding_mode(sa3)
print *,(sqa .eq. sqa3),(ra3 .eq. sa3),(sqa3 .eq. ieee_next_after(sqa, 0.0d0))
lres(10) = (sqa .eq. sqa3) .or. (sqa3 .eq. ieee_next_after(sqa, 0.0d0))
lres(11) = (ra3 .eq. sa3)
lres(12) = (ieee_down .eq. sa3)
lres(13) = (sqa2 .ne. sqa3)

call ieee_set_rounding_mode(ra)
sqa4 = sqrt(a * ieee_value(a,ieee_positive_normal))
call ieee_get_rounding_mode(sa)
print *,(sqa .eq. sqa4), (sa .eq. ra)
lres(14) = (sqa .eq. sqa4)
lres(15) = (sa .eq. ra)

lexp = .true.
call check(lres, lexp, 15)
end
