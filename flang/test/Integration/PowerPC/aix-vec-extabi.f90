! Check that the AIX extended Altivec ABI is emitted as the
! "target-abi" module flag

! REQUIRES: target=powerpc{{.*}}
! RUN: %flang_fc1 -triple powerpc-ibm-aix7.2.0.0 -mabi=vec-extabi -emit-llvm -o - %s | FileCheck %s --check-prefix=EXTABI
! RUN: %flang_fc1 -triple powerpc-ibm-aix7.2.0.0 -mabi=vec-default -emit-llvm -o - %s | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -triple powerpc-ibm-aix7.2.0.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=DEFAULT

! EXTABI: !{i32 1, !"target-abi", !"vec-extabi"}
! DEFAULT-NOT: "target-abi"

subroutine func
end subroutine func
