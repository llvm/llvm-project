! This test checks bind(c) procs inside BLOCK construct.

!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module m
    interface
       subroutine test_proc() bind(C)
       end subroutine test_proc
    end interface
end module m
!CHECK-DAG: %[[S0:.*]] = fir.call @llvm.stacksave.p0() fastmath<contract> : () -> !fir.ref<i8>
!CHECK-DAG: fir.call @test_proc() fastmath<contract> {is_bind_c} : () -> ()
!CHECK-DAG: fir.call @llvm.stackrestore.p0(%[[S0]]) fastmath<contract> : (!fir.ref<i8>) -> ()
!CHECK-DAG: func.func private @test_proc() attributes {fir.bindc_name = "test_proc"}
subroutine test
    BLOCK
        use m
        call test_proc
    END BLOCK
end subroutine test
