** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Optimizer bug in loop processing

	program ka03
      double precision x(10)
      double precision expect(10)
      integer i
      data x/6.00,3.00,2.00,10.00,3.50,7.25,8.90,3.10,1.90,0.70/
      data expect/10.00,8.90,7.25,6.00,3.50,3.10,3.00,2.00,1.90,0.70/
d      write(*,'('' in '',g20.10)')x
      call dsort(x,1,10,i)
d      write(*,'(''out '',g20.10)')x
      call checkd(x, expect, 10)
      end
      subroutine dsort(a,ii,jj,ifail)
      integer           ifail, ii, jj
      double precision  a(jj)
      double precision  t, tt
      integer           i, ij, j, k, l, m
      integer           il(22), iu(22)
      k = 0
      if (ii) 20, 20, 40
   20 continue
cprint *,20
      k = 1
   40 continue
cprint *,40
      if (jj) 600, 600, 60
   60 continue
cprint *,60
      if (k) 620, 80, 620
   80 continue
cprint *,80
      k = 4
      j = ii - jj
      if (j) 100, 640, 620
  100 continue
cprint *,100
      m = 1
      i = ii
      j = jj
  120 continue
cprint *,120
      if (i-j) 140, 400, 400
  140 continue
cprint *,140
      k = i
      ij = (j+i)/2
      t = a(ij)
      if (a(i)-t) 160, 180, 180
  160 continue
cprint *,160
      a(ij) = a(i)
      a(i) = t
      t = a(ij)
  180 continue
cprint *,180
      l = j
      if (a(j)-t) 260, 260, 200
  200 continue
cprint *,200
      a(ij) = a(j)
      a(j) = t
      t = a(ij)
      if (a(i)-t) 220, 260, 260
  220 continue
cprint *,220
      a(ij) = a(i)
      a(i) = t
      t = a(ij)
      go to 260
  240 continue
cprint *,240
      a(l) = a(k)
      a(k) = tt
  260 continue
cprint *,260
      l = l - 1
      if (a(l)-t) 260, 280, 280
  280 continue
cprint *,280
      tt = a(l)
  300 continue
cprint *,300
      k = k + 1
      if (a(k)-t) 320, 320, 300
  320 continue
cprint *,320
      if (k-l) 240, 240, 340
  340 continue
cprint *,340
      if (l-i-j+k) 380, 380, 360
  360 continue
cprint *,360
      il(m) = i
      iu(m) = l
      i = k
      m = m + 1
      go to 440
  380 continue
cprint *,380
      il(m) = k
      iu(m) = j
      j = l
      m = m + 1
      go to 440
  400 continue
cprint *,400
      m = m - 1
      if (m) 420, 640, 420
  420 continue
cprint *,420
      i = il(m)
      j = iu(m)
  440 continue
cprint *,440
      if (j-i-11) 460, 140, 140
  460 continue
cprint *,460
      if (i-ii) 500, 120, 500
  480 continue
cprint *,480
      i = i + 1
  500 continue
cprint *,500
      if (i-j) 520, 400, 520
  520 continue
cprint *,520
      t = a(i+1)
      if (a(i)-t) 540, 480, 480
  540 continue
cprint *,540
      k = i
  560 continue
cprint *,560
      a(k+1) = a(k)
      k = k - 1
      if (t-a(k)) 580, 580, 560
  580 continue
cprint *,580
      a(k+1) = t
      go to 480
  600 continue
cprint *,600
      k = k + 2
  620 continue
cprint *,620
      ifail = ifail
      print *,'k=',k,' ifail=',ifail
      return
  640 continue
cprint *,640
      ifail = 0
      return
      end
