! RUN: %flang_fc1 -triple powerpc64-ibm-aix7.2.0.0 -emit-fir -o - %s | FileCheck %s --check-prefix=FIR
! RUN: %flang_fc1 -triple powerpc64-ibm-aix7.2.0.0 -emit-llvm -O0 -o - %s | FileCheck %s --check-prefix=LLVMIR
! REQUIRES: target=powerpc{{.*}}

! Test that merge() on an array of IBM vector() type lowers correctly through
! a fir.box descriptor at -O0.
!
! At -O0, HLFIR bufferization creates a temporary array and wraps it in a
! fir.box<!fir.array<Nx!fir.vector<...>>>.  Before the fix, both:
!   - getTypeCode()        in FIRType.cpp  hit llvm_unreachable("unsupported type")
!   - getSizeAndTypeCode() in CodeGen.cpp  hit fir::emitFatalError(...)
! because neither handled fir::VectorType as a fir.box element type.
!
! Additionally, getTypeCode() must return CFI_type_struct (not CFI_type_other)
! for fir::VectorType. CFI_type_other is rejected at runtime by
! VerifyEstablishParameters for internal (compiler-generated) descriptors,
! causing a fatal error: "CFI_establish returned 15 for CFI_type_t(-1)".
!
! At -O3, hlfir::createInlineHLFIRCopy inlines the copy element-by-element,
! avoiding the fir.box entirely, which is why -O3 always passed.

subroutine vec_merge_array(va, vb, mask, res)
  vector(integer(4)), intent(in)  :: va(2), vb(2)
  logical,            intent(in)  :: mask(2)
  vector(integer(4)), intent(out) :: res(2)
  res = merge(va, vb, mask)
end subroutine

! FIR-LABEL: func.func @_QPvec_merge_array

! Verify the temporary array and its fir.embox are generated with
! fir.vector<4:i32> as the element type — this is the type that previously
! triggered the ICE in getTypeCode() and getSizeAndTypeCode().
! FIR: fir.allocmem !fir.array<2x!fir.vector<4:i32>>
! FIR: fir.embox {{.*}} : ({{.*}}!fir.array<2x!fir.vector<4:i32>>{{.*}}) -> !fir.box<!fir.array<2x!fir.vector<4:i32>>>

! LLVMIR-LABEL: define void @vec_merge_array_
! Verify CodeGen completes and produces correct vector stores.
! LLVMIR: store <4 x i32>
