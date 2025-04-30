!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p
  
  parameter(NbrTests=1680)
  parameter(o_extent=2)
  parameter(n_extent=6)
  parameter(m_extent=4)
  parameter(k_extent=8)
  
  logical*4, dimension(n_extent,m_extent) :: arr1
  logical*4, dimension(m_extent,k_extent) :: arr2
  logical*4, dimension(n_extent,k_extent) :: arr3
  logical*4, dimension(n_extent,m_extent,o_extent) :: arr4
  logical*4, dimension(n_extent,o_extent,m_extent) :: arr5
  logical*4, dimension(o_extent,n_extent,m_extent) :: arr6
  
  logical*4, dimension(o_extent,m_extent,k_extent) :: arr7
  logical*4, dimension(m_extent,o_extent,k_extent) :: arr8
  logical*4, dimension(m_extent,k_extent,o_extent) :: arr9
  
  logical*4, dimension(n_extent,k_extent,o_extent) :: arr10
  logical*4, dimension(n_extent,o_extent,k_extent) :: arr11
  logical*4, dimension(o_extent,n_extent,k_extent) :: arr12
  
  logical*4, dimension(2:n_extent+1,2:m_extent+1) :: arr13
  logical*4, dimension(2:m_extent+1,2:k_extent+1) :: arr14
  logical*4, dimension(2:n_extent+1,2:k_extent+1) :: arr15

  logical*4, dimension(n_extent,k_extent) :: arr16
  
  logical*4 :: expect(NbrTests) 
  logical*4 :: results(NbrTests)
  
  integer:: i,j
  
  data arr1 /.true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false./
  data arr2 /.false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true./
  data arr3 /.true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true./
  data arr4 /.false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
             .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false./
  data arr5 / .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false.,  &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true./
  data arr6 /.true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false.,  &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true./
  data arr7 /.true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .false.,.true.,.false.,.true.,.false.,.true.,.false.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .false.,.true.,.false.,.true.,.false.,.true.,.false.,.true./
  data arr8 /.true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false.,  &
             .true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true./
  data arr9 /.true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false.,  &
             .false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .true.,.false.,.true.,.false.,.true.,.false.,.true.,.false., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true./
  data arr10 /.false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true./
  data arr11 /.true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false./
  data arr12 /.true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
             .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.true.,.false., &
             .false.,.false.,.false.,.true.,.true.,.true.,.false.,.true., &
	     .true.,.true.,.true.,.false.,.false.,.false.,.true.,.false., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false.,  &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .false.,.false.,.true.,.true.,.true.,.true.,.false.,.false., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true., &
             .false.,.false.,.true.,.true.,.false.,.false.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false./
  data arr13 /.true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false./ 
  data arr14 /.false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .true.,.true.,.false.,.false.,.true.,.true.,.false.,.false., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true., &
             .false.,.false.,.false.,.false.,.true.,.true.,.true.,.true./ 
  data arr15 /.true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true., &
             .true.,.true.,.true.,.true.,.true.,.true.,.true.,.true./
  
  data expect /   &
  ! test 1-48
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, &
     -1, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, -1, &
  ! test 49-96
      0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, &
     -1, 0, 0, -1, -1, 0, -1, 0, 0, -1, -1, 0, 0, &
      0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, &
  ! test 97-144
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, &
     -1, 0, 0, -1, 0, -1, -1, 0, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, 0, &
  ! test 145-192
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, -1, &
  ! test 193-240, &
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, -1, &
  ! test 241-288
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, &
     -1, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, -1, &
  ! test 289-336
      0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, &
     -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 337-384
      0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 385-432
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, &
     -1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 433-480, &
      0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, &
      0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, 0, &
      0, 0, 0, -1, 0, -1, 0, -1, 0, &
  ! test 481-528
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 529-576
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 577-624
      0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, &
      0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, &
      0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, -1, 0, -1, &
  ! test 625-672
      0, -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, &
     -1, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, &
     -1, 0, -1, 0, 0, 0, 0, 0, 0, &
  ! test 673-720, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, &
     -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, &
  ! test 721-768
      0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, &
      0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, &
      0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, -1, 0, -1, &
  ! test 769-864
      0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, &
     -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 865-960, &
      0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, &
     -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 961-1056
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, &
      0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
      0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, &
     -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, &
      0, 0, 0, 0, 0, &
  ! test 1057-1152
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 1153-1248
      0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
     -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 1249-1344
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, &
      0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, &
      0, 0, 0, -1, 0, &
  ! test 1345-1440, &
      0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, &
      0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, &
     -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 1441-1536
      0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, &
  ! test 1537-1632
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, &
      0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
      0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, &
     -1, 0, 0, 0, -1, & 
  ! test 1633-1680 
      0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, &
     -1, 0, 0, -1, -1, -1, -1, 0, 0, -1, -1, 0, 0, &
      0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, &
      0, 0, 0, -1, -1, -1, -1, -1, -1/
  
  ! test 1-48
  arr3=0
  arr3 = matmul(arr1,arr2)
  call assign_result(1,48,arr3,results)
  !print *,"test 1-48"
  !print *,arr3
  
  ! test 49-96
  arr3=0
  arr3(2:n_extent,:) = matmul(arr1(2:n_extent,:),arr2)
  call assign_result(49,96,arr3,results)
  !print *,"test 49-96"
  !print *,arr3
  
  ! test 97-144
  arr3=0
  arr3(1:n_extent-1,:) = matmul(arr1(1:n_extent-1,:),arr2)
  call assign_result(97,144,arr3,results)
  !print *,"test 97-144"
  !print *,arr3
  
  ! test 145-192
  arr3=0
  arr3 = matmul(arr1(:,2:m_extent),arr2(2:m_extent,:))
  call assign_result(145,192,arr3,results)
  !print *,"test 145-192"
  !print *,arr3
  
  ! test 193-240
  arr3=0
  arr3 = matmul(arr1(:,2:m_extent),arr2(2:m_extent,:))
  call assign_result(193,240,arr3,results)
  !print *,"test 193-240"
  !print *,arr3
  
  ! test 241-288
  arr3=0
  arr3 = matmul(arr1(:,1:m_extent-1),arr2(1:m_extent-1,:))
  call assign_result(241,288,arr3,results)
  !print *,"test 241-288"
  !print *,arr3
  
  ! test 289-336
  arr3=0
  arr3(1:3,1:3) = matmul(arr1(1:3,1:3),arr2(1:3,1:3))
  call assign_result(289,336,arr3,results)
  !print *,"test 289-336"
  !print *,arr3
  
  ! test 337-384
  arr3=0
  arr3(2:4,2:4) = matmul(arr1(2:4,2:4),arr2(2:4,2:4))
  call assign_result(337,384,arr3,results)
  !print *,"test 337-384"
  !print *,arr3
  
  ! test 385-432
  arr3=0
  arr3(:,1:k_extent:2) = matmul(arr1(:,1:m_extent-1),arr2(1:m_extent-1,1:k_extent:2))
  call assign_result(385,432,arr3,results)
  !print *,"test 385-432"
  !print *,arr3
  
  ! test 433-480
  arr3=0
  arr3(1:n_extent:2,:) = matmul(arr1(1:n_extent:2,2:m_extent),arr2(1:m_extent-1,:))
  call assign_result(433,480,arr3,results)
  !print *,"test 433-480"
  !print *,arr3
  
  ! test 481-528
  arr3=0
  arr3(1:n_extent:2,1:k_extent:2) = matmul(arr1(1:n_extent:2,1:m_extent-1),      &
                                           arr2(1:m_extent-1,1:k_extent:2))
  call assign_result(481,528,arr3,results)
  !print *,"test 481-528"
  !print *,arr3
  
  ! test 529-576
  arr3=0
  arr3(1:n_extent-1:2,1:k_extent-1:2) = matmul(arr1(1:n_extent-1:2,2:m_extent),	&
                                               arr2(1:m_extent-1,1:k_extent:2))
  call assign_result(529,576,arr3,results)
  !print *,"test 529-576"
  !print *,arr3
  
  ! test 577-624
  arr3=0
  arr3(2:n_extent:2,2:k_extent:2) = matmul(arr1(2:n_extent:2,1:m_extent-1),	&
                                               arr2(2:m_extent,2:k_extent:2))
  call assign_result(577,624,arr3,results)
  !print *,"test 577-624"
  !print *,arr3
  
  ! test 625-672
  arr3=0
  arr3(n_extent:1:-2,1:k_extent:2) = matmul(arr1(n_extent:1:-2,1:m_extent-1),      &
                                           arr2(1:m_extent-1,k_extent:1:-2))
  call assign_result(625,672,arr3,results)
  !print *,"test 625-672"
  !print *,arr3
  
  ! test 673-720
  arr3=0
  arr3(1:n_extent-1:2,k_extent-1:1:-2) = matmul(arr1(1:n_extent-1:2,m_extent:2:-1),	&
                                               arr2(m_extent-1:1:-1,1:k_extent:2))
  call assign_result(673,720,arr3,results)
  !print *,"test 673-720"
  !print *,arr3
  
  ! test 721-768
  arr3=0
  arr3(n_extent:2:-2,k_extent:2:-2) = matmul(arr1(n_extent:2:-2,m_extent-1:1:-1),	&
                                               arr2(m_extent:2:-1,k_extent:2:-2))
  call assign_result(721,768,arr3,results)
  !print *,"test 721-768"
  !print *,arr3
  
  ! test 769-864
  arr10=0
  arr10(2:4,2:4:1) = matmul(arr4(2:4,2:4,1),arr7(1,2:4,2:4))
  call assign_result(769,864,arr10,results)
  !print *,"test 769-864"
  !print *,arr10
  
  ! test 865-960
  arr11=0
  arr11(:,1,1:k_extent:2) = matmul(arr4(:,1:m_extent-1,2),arr8(1:m_extent-1,1,1:k_extent:2))
  call assign_result(865,960,arr11,results)
  !print *,"test 865-960"
  !print *,arr11
  
  ! test 961-1056
  arr12=0
  arr12(2,1:n_extent:2,:) = matmul(arr4(1:n_extent:2,2:m_extent,2),arr9(1:m_extent-1,:,2))
  call assign_result(961,1056,arr12,results)
  !print *,"test 961-1056"
  !print *,arr12
  
  ! test 1057-1152
  arr10=0
  arr10(1:n_extent:2,1:k_extent:2,2) = matmul(arr5(1:n_extent:2,2,1:m_extent-1),      &
                                           arr8(1:m_extent-1,2,1:k_extent:2))
  call assign_result(1057,1152,arr10,results)
  !print *,"test 1057-1152"
  !print *,arr10
  
  ! test 1153-1248
  arr11=0
  arr11(1:n_extent-1:2,2,1:k_extent-1:2) = matmul(arr5(1:n_extent-1:2,1,2:m_extent),	&
                                               arr9(1:m_extent-1,1:k_extent:2,1))
  call assign_result(1153,1248,arr11,results)
  !print *,"test 1153-1248"
  !print *,arr11
  
  ! test 1249-1344
  arr12=0
  arr12(1,2:n_extent:2,2:k_extent:2) = matmul(arr5(2:n_extent:2,2,1:m_extent-1),	&
                                               arr7(2,2:m_extent,2:k_extent:2))
  call assign_result(1249,1344,arr12,results)
  !print *,"test 1249-1344"
  !print *,arr12
  
  ! test 1345-1440
  arr10=0
  arr10(n_extent:1:-2,1:k_extent:2,1) = matmul(arr6(2,n_extent:1:-2,1:m_extent-1),      &
                                           arr9(1:m_extent-1,k_extent:1:-2,2))
  call assign_result(1345,1440,arr10,results)
  !print *,"test 1345-1440"
  !print *,arr10
  
  ! test 1441-1536
  arr11=0
  arr11(1:n_extent-1:2,2,k_extent-1:1:-2) = matmul(arr6(1,1:n_extent-1:2,m_extent:2:-1),	&
                                               arr7(2,m_extent-1:1:-1,1:k_extent:2))
  call assign_result(1441,1536,arr11,results)
  !print *,"test 1441-1536"
  !print *,arr11
  
  ! test 1537-163
  arr12=0
  arr12(2,n_extent:2:-2,k_extent:2:-2) = matmul(arr6(2,n_extent:2:-2,m_extent-1:1:-1),	&
                                               arr8(m_extent:2:-1,1,k_extent:2:-2))
  call assign_result(1537,1632,arr12,results)
  !print *,"test 1537-1632"
  !print *,arr12

  arr16 = .false.
  
  ! test 1663-1680
  arr15=0 
  arr15 = arr16 .or. matmul(arr13,arr14)
  call assign_result(1633,1680,arr15,results)
  !print *,"test 1663-1680"
  !print *,arr15

  call check(results, expect, NbrTests)
end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  logical*4, dimension(1:e_idx-s_idx+1) :: arr
  logical*4, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine
