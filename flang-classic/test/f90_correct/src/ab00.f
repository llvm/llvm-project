** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Test input line format.  This includes comments, continuation lines,
*   long lines, tabs in lines, and the label field.

     0program
     \p
      
c     next line consists of >100 blanks:
                                                                                                            
      integer rslts(6), e x p e c t(6        
&     )

C     following comment lines have varying numbers of blanks after "c":
c
c 
c    
c                                      
C     nineteen continuation lines:
C       data expect /11, 21, 31, 41, 51, 61 /

        data
c       comments should be allowed within continuations.
     "expect
     //
&     1
     -  1
     1,
     2  21
     3            ,
&     31
     9,                                                                     
Cmore comments ...


     +4                           1
     *,
c  -- some empty continuation lines:
     +
&
     (
     ) 51
c           fifteen continuations so far.
     &                 
     +,
     =61
     a       /

C    test labels:

      goto 0010
00010 rslts(1) = 11
      goto 9 9 9 9 9
    1 rslts(2) = 21
      goto 00002
c   next two lines have tab character after label:
99999	goto 1
  2	continue

C    check that line length is 72 and characters after 72 are ignored:

      rslts(3) =                                                      319ZZZZZZZ
      rslts(4) =                                                       4000
     +                                                                 01xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c      -- line exactly 72 characters:
      rslts(4) = rslts(4) +                                            1

C    test tabs in lines:

	rslts	(	5) = 5	1
		r	slts(6	)
&	= 61

C    test that END statement is recognized correctly:

      call check1(r s l t s, e x p e c t, 6)
        e  n  d
c - comment between subprograms
      subroutine check1(r, e, i)
      call check2(r, e, i)
	E  N  D

      subroutine
     +             check2(r, e, i)
      call check(r, e, i)
      END
c  comments after last END statement should not cause fatal error.
