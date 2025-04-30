!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


          subroutine initialize(a)
          character*5 a(10,10)
          a(1,:) = "jbc"
          a(2,:) = "def"
          a(3,:) = "gh"
          a(4,:) = "ij"
          a(5,:) = "kj"
          a(6,:) = "abc"
          a(7,:) = "op"
          a(8,:) = "qr"
          a(9,:) = "st"
          a(10,:) = "ug"

          end subroutine

          program mymax
          parameter(N=86)
          integer result(86), expect(N)
          character*5 a(10,10)
          data expect /3,2,6,6,6,6,6,6,6,6,&
                       &6,6,1,1,1,1,1,1,1,1,&
                       &1,1,10,10,10,10,10,10,10,10,&
                       &10,10,1,1,1,1,1,1,1,1,&
                       &1,1,2,1,10,1,2,2,2,2,&
                       &2,2,2,2,2,2,1,1,1,1,&
                       &1,0,1,1,1,1,10,10,10,10,&
                       &10,10,10,10,10,10,1,1,1,1,&
                       &1,0,1,1,1,1 /

          call initialize(a)

! test basic min/maxloc
          result(1:1) = minloc((/'j','d','a'/))
          result(2:2) = maxloc((/' dk','mf','ah'/))

! test dim
          result(3:12) = minloc(a, dim=1)
          result(13:22) = minloc(a, dim=2)
          result(23:32) = maxloc(a, dim=1)
          result(33:42) = maxloc(a, dim=2)
          

! test mask
          result(43:44) = minloc(a, mask = a > 'bc')
          result(45:46) = maxloc(a, mask = a > 'br')


! test mask with dim
          result(47:56) = minloc(a, mask = a > 'bc', dim=1)
          result(57:66) = minloc(a, mask = a > 'bc', dim=2)
          result(67:76) = maxloc(a, mask = a > 'br', dim=1)
          result(77:86) = maxloc(a, mask = a > 'br', dim=2)


          call check(result, expect, N)

          end
