! This test checks bind(c) procs inside BLOCK construct.

!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m
    interface
       subroutine test_proc() bind(C)
       end subroutine test_proc
    end interface
end module m
!CHECK-DAG: %[[S0:.*]] = llvm.intr.stacksave : !llvm.ptr
!CHECK-DAG: fir.call @test_proc() proc_attrs<bind_c> fastmath<contract> : () -> ()
!CHECK-DAG: llvm.intr.stackrestore %[[S0]] : !llvm.ptr
!CHECK-DAG: func.func private @test_proc() attributes {fir.bindc_name = "test_proc"}
subroutine test
    BLOCK
        use m
        call test_proc
    END BLOCK
end subroutine test
