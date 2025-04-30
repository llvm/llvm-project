** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   PARAMETER statements for LOGICAL constants.  Also,
*   constant folding of logical  expressions.

      implicit logical (a, o)
      logical t1, t2, f1, f2
      parameter (t1 = .true.)
      parameter (f1 = .false., t2 = .not..false., f2 = .not.t1)

      parameter (a1 = f1 .and. .false.,
     +           a2 = a1 .and. .true. ,
     +           a3 = .true..and..false.,
     +           a4 = .true..and..true.  )

      parameter (o1 = f1 .or. f1,
     +           o2 = f1 .or. t1,
     +           o3 = t1 .or. f1,
     +           o4 = t1 .or. t1  )

      logical e1, e2, e3, e4, n1, n2, n3, n4
      parameter (e1 = f1 .eqv. f1,
     +           e2 = f1 .eqv. t1,
     +           e3 = t1 .eqv. f1,
     +           e4 = t1 .eqv. t1  )

      parameter (n1 = f1 .neqv. f1,
     +           n2 = f1 .neqv. t1,
     +           n3 = t1 .neqv. f1,
     +           n4 = t1 .neqv. t1  )

      logical x1, x2, x3, x4
      parameter(N = 24)
      logical rslts(N), expect(N)
      parameter( x1 = (.true. .and. .false.) .or. (.false. .eqv. f1) )
      parameter(x2  = 3 .gt. 4 .or. 6 .eq. 1)
      parameter(x3 = .not.f1.and.t1  .neqv.  f1.or..false..or..false.)
      parameter(x4 = .not. (2 .le. 3 .eqv. 78 .eq. 78) )
      
      data expect / .true., .false., .true., .false.,
c tests 5 - 8:   AND operation
     +              .false., .false., .false., .true.,
c tests 9 - 12:  OR operation
     +              .false., .true., .true., .true.,
c tests 13 - 16: EQV operation
     +              .true., .false., .false., .true.,
c tests 17 - 20: NEQV operation
     +              .false., .true., .true., .false.,
c tests 21 - 24: miscellaneous combinations
     +              .true., .false., .true., .false.   /

      data (rslts(i), i = 1, 16) / t1, f1, t2, f2,
     +                             a1, a2, a3, a4,
     +                             o1, o2, o3, o4,
     +                             e1, e2, e3, e4  /

      data (rslts(i), i = 17, 20)/ n1, n2, n3, n4  /

      rslts(21) = x1
      rslts(22) = x2
      rslts(23) = x3
      rslts(24) = x4
      if (x4)   rslts(24) = .true.

      call check(rslts, expect, N)
      end
