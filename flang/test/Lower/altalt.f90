! RUN: bbc -emit-fir -o - %s | FileCheck %s

   ! CHECK-LABEL: func @_QPss
   subroutine ss(n)
     print*, n
     ! CHECK: return{{$}}
     return
   ! CHECK-LABEL: func @_QPee
   entry ee(n,*)
     ! CHECK: return %{{.}} : index
     return 1
   end

   ! CHECK-LABEL: func @_QQmain
     call ss(7)
     call ee(2, *11)
     print*, 'default'
11   print*, 11
   end
