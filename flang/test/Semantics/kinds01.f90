! REQUIRES: x86_64-registered-target
! RUN: %python %S/test_symbols.py %s %flang_fc1
 !DEF: /MainProgram1/jk1 ObjectEntity INTEGER(1)
 integer(kind=1) jk1
 !DEF: /MainProgram1/js1 ObjectEntity INTEGER(1)
 integer*1 js1
 !DEF: /MainProgram1/jk2 ObjectEntity INTEGER(2)
 integer(kind=2) jk2
 !DEF: /MainProgram1/js2 ObjectEntity INTEGER(2)
 integer*2 js2
 !DEF: /MainProgram1/jk4 ObjectEntity INTEGER(4)
 integer(kind=4) jk4
 !DEF: /MainProgram1/js4 ObjectEntity INTEGER(4)
 integer*4 js4
 !DEF: /MainProgram1/jk8 ObjectEntity INTEGER(8)
 integer(kind=8) jk8
 !DEF: /MainProgram1/js8 ObjectEntity INTEGER(8)
 integer*8 js8
 !DEF: /MainProgram1/jk16 ObjectEntity INTEGER(16)
 integer(kind=16) jk16
 !DEF: /MainProgram1/js16 ObjectEntity INTEGER(16)
 integer*16 js16
 !DEF: /MainProgram1/ak2 ObjectEntity REAL(2)
 real(kind=2) ak2
 !DEF: /MainProgram1/as2 ObjectEntity REAL(2)
 real*2 as2
 !DEF: /MainProgram1/ak4 ObjectEntity REAL(4)
 real(kind=4) ak4
 !DEF: /MainProgram1/as4 ObjectEntity REAL(4)
 real*4 as4
 !DEF: /MainProgram1/ak8 ObjectEntity REAL(8)
 real(kind=8) ak8
 !DEF: /MainProgram1/as8 ObjectEntity REAL(8)
 real*8 as8
 !DEF: /MainProgram1/dp ObjectEntity REAL(8)
 double precision dp
 !DEF: /MainProgram1/ak10 ObjectEntity REAL(10)
 real(kind=10) ak10
 !DEF: /MainProgram1/as10 ObjectEntity REAL(10)
 real*10 as10
 !DEF: /MainProgram1/ak16 ObjectEntity REAL(16)
 real(kind=16) ak16
 !DEF: /MainProgram1/as16 ObjectEntity REAL(16)
 real*16 as16
 !DEF: /MainProgram1/zk2 ObjectEntity COMPLEX(2)
 complex(kind=2) zk2
 !DEF: /MainProgram1/zs2 ObjectEntity COMPLEX(2)
 complex*4 zs2
 !DEF: /MainProgram1/zk4 ObjectEntity COMPLEX(4)
 complex(kind=4) zk4
 !DEF: /MainProgram1/zs4 ObjectEntity COMPLEX(4)
 complex*8 zs4
 !DEF: /MainProgram1/zk8 ObjectEntity COMPLEX(8)
 complex(kind=8) zk8
 !DEF: /MainProgram1/zs8 ObjectEntity COMPLEX(8)
 complex*16 zs8
 !DEF: /MainProgram1/zdp ObjectEntity COMPLEX(8)
 double complex zdp
 !DEF: /MainProgram1/zk10 ObjectEntity COMPLEX(10)
 complex(kind=10) zk10
 !DEF: /MainProgram1/zs10 ObjectEntity COMPLEX(10)
 complex*20 zs10
 !DEF: /MainProgram1/zk16 ObjectEntity COMPLEX(16)
 complex(kind=16) zk16
 !DEF: /MainProgram1/zs16 ObjectEntity COMPLEX(16)
 complex*32 zs16
 !DEF: /MainProgram1/lk1 ObjectEntity LOGICAL(1)
 logical(kind=1) lk1
 !DEF: /MainProgram1/ls1 ObjectEntity LOGICAL(1)
 logical*1 ls1
 !DEF: /MainProgram1/lk2 ObjectEntity LOGICAL(2)
 logical(kind=2) lk2
 !DEF: /MainProgram1/ls2 ObjectEntity LOGICAL(2)
 logical*2 ls2
 !DEF: /MainProgram1/lk4 ObjectEntity LOGICAL(4)
 logical(kind=4) lk4
 !DEF: /MainProgram1/ls4 ObjectEntity LOGICAL(4)
 logical*4 ls4
 !DEF: /MainProgram1/lk8 ObjectEntity LOGICAL(8)
 logical(kind=8) lk8
 !DEF: /MainProgram1/ls8 ObjectEntity LOGICAL(8)
 logical*8 ls8
end program
