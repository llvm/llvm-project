! RUN: %if !target={{.*aix.*}} %{ \
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=5 %s  \
! RUN:         -o - | FileCheck --check-prefix=CHECK-DWARF5 %s \
! RUN: %}

! RUN: %if !target={{.*aix.*}} %{ \
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only -dwarf-version=5 \
! RUN:         %s -o - | FileCheck --check-prefix=CHECK-DWARF5 %s \
! RUN: %}

! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=4 %s  \
! RUN:         -o - | FileCheck --check-prefix=CHECK-DWARF4 %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=3 %s  \
! RUN:         -o - | FileCheck --check-prefix=CHECK-DWARF3 %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=2 %s  \
! RUN:         -o - | FileCheck --check-prefix=CHECK-DWARF2 %s
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s  -o -\
! RUN:         | FileCheck --check-prefix=CHECK-WITHOUT-VERSION %s
! RUN: %flang_fc1 -emit-llvm -dwarf-version=5 %s  -o - \
! RUN:         | FileCheck --check-prefix=CHECK-WITHOUT-VERSION %s

program test
end program test

! CHECK-DWARF5: !{i32 7, !"Dwarf Version", i32 5}
! CHECK-DWARF4: !{i32 7, !"Dwarf Version", i32 4}
! CHECK-DWARF3: !{i32 7, !"Dwarf Version", i32 3}
! CHECK-DWARF2: !{i32 7, !"Dwarf Version", i32 2}
! CHECK-WITHOUT-VERSION-NOT: "Dwarf Version"
