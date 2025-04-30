** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS UNION/RECORD

c  test nested union references

	program br30
	parameter (N=15)
	integer expect(N), result(N)

      structure /glob_geom_par/
	integer arr1(1)
      endstructure
      structure /vdc_plane/
	integer arr2(2)
      endstructure
      structure /tracks/
	integer arr3(3)
      endstructure
      structure /vdc_package/
         union
            map
               integer ntrack,    !  number of tracks
     +                 tx_cor     !  t->x conversion correction method
               record /glob_geom_par/ geom
               union
                  map
                     record /vdc_plane/ u1,v1,u2,v2
                  end map
                  map
                     record /vdc_plane/ w(3)
                  end map
               end union
               integer iord(3) !    array giving ascending order of
                                      !     t0_av for tracks
               record /tracks/ track(3)
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
c print a few members
d      print 99, vd.ntrack
d      print 99, vd.tx_cor
d      print 99, vd.geom.arr1(1)
d      print 99, vd.u1.arr2(1)
d      print 99, vd.v1.arr2(2)
d      print 99, vd.u2.arr2(1)
d      print 99, vd.v2.arr2(2)
d      print 99, vd.w(1).arr2(2)
d      print 99, vd.w(2).arr2(1)
d      print 99, vd.w(3).arr2(2)
d      print 99, vd.iord(2)
d      print 99, vd.iord(3)
d      print 99, vd.track(1).arr3(1)
d      print 99, vd.track(2).arr3(2)
d      print 99, vd.track(3).arr3(3)
d99    format(1x,i3)

	result(1) = vd.ntrack
	result(2) = vd.tx_cor
	result(3) = vd.geom.arr1(1)
	result(4) = vd.u1.arr2(1)
	result(5) = vd.v1.arr2(2)
	result(6) = vd.u2.arr2(1)
	result(7) = vd.v2.arr2(2)
	result(8) = vd.w(1).arr2(2)
	result(9) = vd.w(2).arr2(1)
	result(10) = vd.w(3).arr2(2)
	result(11) = vd.iord(2)
	result(12) = vd.iord(3)
	result(13) = vd.track(1).arr3(1)
	result(14) = vd.track(2).arr3(2)
	result(15) = vd.track(3).arr3(3)

	call check (result, expect, N)

	data expect /
     +   1, 2, 3, 4, 7,		! results( 1- 5)
     +   8, 11, 5, 6, 9,	! results( 6-10)
     +   13, 14, 15, 19, 23	! results(11-15)
     +  /

	end
