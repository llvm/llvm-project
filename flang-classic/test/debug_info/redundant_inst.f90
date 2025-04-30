!RUN: %flang %s -g -S -emit-llvm -o - | FileCheck %s --check-prefix=STORE
!RUN: %flang %s -g -gdwarf-4 -S -emit-llvm -o - | llc -O0 -fast-isel=true -filetype=obj -o %t
!RUN: llvm-dwarfdump --debug-line %t | FileCheck %s --check-prefix=LINETABLE

!Check that `store` instruction is getting emitted for the second assignment.
!STORE: store i32 4, ptr %[[VAR_A:.*]], align 4
!STORE: %[[TEMP:.*]] = load i32, ptr %[[VAR_A]], align 4, !dbg ![[LOCATION:.*]]
!STORE: store i32 %[[TEMP]], ptr %[[VAR_A]], align 4, !dbg ![[LOCATION]]
!STORE: ![[LOCATION]] = !DILocation(line: 19, column: 1, scope: !{{.*}})

!Check the line table entry of the second assignment.
!LINETABLE: Address    Line Column File ISA Discriminator {{(OpIndex )?}}Flags
!LINETABLE: 0x{{.*}}    19    1      1   0         0      {{(0 )?}}is_stmt


program main
       integer :: a
       a = 4
       a = a !line no. 19
       print*, a
end
