
!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!  Test the values returned by IO statement specifiers (esp. when 
!  compiled with `-i8').

       PROGRAM na25
       
       parameter(N=18)
       integer :: results(N)
       integer :: expected(N)
       
       character(len=20) :: line = 'abcdefghij'
       
       integer :: a(2)
       CHARACTER*10 B10VK, C10VK, E11VK*11, G10VK
       integer :: ios, nxtrec, nbr, unt
       logical :: nm, opn, exst
       
       data results / 0, 0, 0, -1, -1, -1, &
                      0, 0, 0, -1, -1, -1, &
                      -1, &
                      -1, &
                      -1, &
                      0, &
                      -1, &
                      -1/
       data expected / -1, -1, -1, 0, 0, 5, &
                       -1, -1, -1, 0, 0, 6, &
                       209, &
                       0, &
                       0, &
                       -1, &
                       0, &
                       0/

       
       INQUIRE (5,EXIST=exst,NAMED=nm, OPENED=opn,IOSTAT=ios,NEXTREC=nxtrec, &
                  NUMBER=nbr)
       results(1)=exst
       results(2)=nm
       results(3)=opn
       results(4)=ios
       results(5)=nxtrec
       results(6)=nbr
       
!       print *,results(1:6)
       
       INQUIRE (6,EXIST=exst,NAMED=nm, OPENED=opn,IOSTAT=ios,NEXTREC=nxtrec, &
                  NUMBER=nbr)
       results(7) = exst
       results(8)=nm
       results(9)=opn
       results(10)=ios
       results(11)=nxtrec
       results(12)=nbr

!       print *,results(7:12)

       open(8, FILE='NonExistent', STATUS='OLD', IOSTAT=ios)
       results(13) = ios
!       print *,results(13)

       open(8, FILE='na25dat.out', STATUS='REPLACE', FORM='UNFORMATTED', &
               ACCESS='SEQUENTIAL', ACTION='READWRITE', IOSTAT=ios)
       results(14) = ios
!       print *,results(14)
  
       write(8, IOSTAT=ios) line
       results(15) = ios
!       print *,results(15)

       line = '                    '

       read(8, IOSTAT=ios) line
       results(16) = ios
!       print *,results(16)

       rewind(8, IOSTAT=ios)
       results(17) = ios
!       print *,results(17)

       read(8, IOSTAT=ios) line
       results(18) = ios
!       print *,results(18)


       call check(results, expected, 18)

       END PROGRAM na25
