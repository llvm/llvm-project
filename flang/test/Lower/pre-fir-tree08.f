! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenacc %s | FileCheck %s
	program rewrite_goto
	integer b
	
	b = dummy(10)

	end
      function dummy(a)
      integer, a
      
      do 10 i=1,10
  10  if(i .EQ. 1) GOTO 11
      i=0
  11  dummy = a + i
      return
      end

! CHECK: <<IfConstruct!>> -> 5
! CHECK: 2 ^IfStmt -> 5: 10if(i.eq.1)goto11
! CHECK: 3 ^GotoStmt! -> 7: goto11

