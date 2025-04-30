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
          character*5 char_result(86)
          character*5 a(10,10)
          data expect /-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,0,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,&
                       &-1,0,-1,-1,-1,-1 /

          call initialize(a)

! test basic min/maxloc
! output "a", "mf"
          char_result(1:1) = minval((/'j','d','a'/))
          char_result(2:2) = maxval((/' dk','mf','ah'/))
          result(1) = (char_result(1) == 'a')
          result(2) = (char_result(2) == 'mf')

! test dim
          char_result(3:12) = minval(a, dim=1)
          char_result(13:22) = minval(a, dim=2)
          char_result(23:32) = maxval(a, dim=1)
          char_result(33:42) = maxval(a, dim=2)
          result(3:12) = (char_result(3:12) == a(6,:))
          result(13:22) = (char_result(13:22) == a(:,1))
          result(23:32) = (char_result(23:32) == a(10,:))
          result(33:42) = (char_result(33:42) == a(:,1))
          
! test mask
          char_result(43:44) = minval(a, mask = a > 'bc')
          char_result(45:46) = maxval(a, mask = a > 'br')
          result(43:44) = (char_result(43:44) == a(2,1:2) )
          result(45:46) = (char_result(45:46) == a(10,1:2))

! test mask with dim
          char_result(47:56) = minval(a, mask = a > 'bc', dim=1)
          char_result(57:66) = minval(a, mask = a > 'bc', dim=2)
          char_result(67:76) = maxval(a, mask = a > 'br', dim=1)
          char_result(77:86) = maxval(a, mask = a > 'br', dim=2)
          result(47:56) = (char_result(47:56) == a(2,:) )
          result(57:66) = (char_result(57:66) == a(:,1))
          result(67:76) = (char_result(67:76) == a(10,:) )
          result(77:86) = (char_result(77:86) == a(:,1))

          call check(result, expect, N)

          end
