! Allocating the dynamically sized automatic variables in unified or managed
! memory is a correctness requirement of -gpu=mem:unified|managed, so the pass
! doing it stays in the pipeline even where the array allocation optimization
! is disabled.

! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -mmlir -disable-memory-allocation-opt -o /dev/null %s 2>&1 | FileCheck %s

! REQUIRES: asserts

end program

! CHECK: CudaHeapAllocPromotion
! CHECK-NOT: MemoryAllocationOpt
