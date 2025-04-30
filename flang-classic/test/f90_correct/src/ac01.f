** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Compiler directives: %list, %nolist, %eject. include statement

C   NONSTANDARD:
C     VMS directives %LIST, %NOLIST, %ELECT, and VMS INCLUDE stmt.
C     INCLUDE statement may span several lines

%list
%LIST  
	common /c/ rslts(6), expect(6)
	integer rslts, expect
	data expect / 1, 2, 3, 4, 5, 6/

%nolist
C

	rslts(1)
&          = 1
%list
	Include '
     +  a
     +  c
     +  0
     +  0
     +	.	
     +  i
     +  /
     +  n
     +  o
     +  l
     +  i
     +  s
     +  t
     +  '
	INCLUDE '
     +  a
     +  c
     +  0
     +  0
     +	.	
     +  h
     +  /
     +  l
     +  i
     +  s
     +  t
     +  '
%eject
	rslts(5) = 5
	rslts(6) = if(5)
	rslts(3) = 3
	call check(rslts, expect, 6)
	end
c  comment lines between two subprograms ...

	include
     +  '
     +  a
     +  c
     +  0
     +  0
     +	.	h3'
%nolist
