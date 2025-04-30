** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS UNION/RECORD

c  test nested structures references

	program br31
	parameter (N=8)
	integer expect(N), result(N)

      structure /vdc_package/
         union
            map
		integer x
		structure /foobar/ rr
		    integer m1
		    integer m2
		endstructure
		integer y
		record /foobar/ss
		structure uu
		    integer n1
		    integer n2
		endstructure
            end map
            map
               integer var(100)
            end map
         end union
      end structure
      record /vdc_package/ vd
      common /vd/vd
      do i = 1, 100
	  vd.var(i) = i
      enddo
d99    format(1x,i3)
d      print 99, vd.x
d      print 99, vd.rr.m1
d      print 99, vd.rr.m2
d      print 99, vd.y
d      print 99, vd.ss.m1
d      print 99, vd.ss.m2
d      print 99, vd.uu.n1
d      print 99, vd.uu.n2

	result(1) = vd.x
	result(2) = vd.rr.m1
	result(3) = vd.rr.m2
	result(4) = vd.y
	result(5) = vd.ss.m1
	result(6) = vd.ss.m2
	result(7) = vd.uu.n1
	result(8) = vd.uu.n2

	call check (result, expect, N)

	data expect /
     +   1, 2, 3, 4, 5,		! results( 1- 5)
     +   6, 7, 8		! results( 6- 8)
     +  /

	end
