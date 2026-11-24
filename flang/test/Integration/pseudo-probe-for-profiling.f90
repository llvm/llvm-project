! Test -fpseudo-probe-for-profiling option runs SampleProfileProbePass and emits llvm.pseudoprobe intrinsic calls.
!
! RUN: %flang_fc1 -emit-llvm -fdebug-pass-manager -fpseudo-probe-for-profiling -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=PROBE-PASS
! RUN: %flang_fc1 -emit-llvm -O0 -fpseudo-probe-for-profiling -o - %s | FileCheck %s --check-prefix=PROBE
! RUN: %flang_fc1 -emit-llvm -O2 -fpseudo-probe-for-profiling -o - %s | FileCheck %s --check-prefix=PROBE

! Test that -fdebug-info-for-profiling combined with -fpseudo-probe-for-profiling still emits pseudo-probes and debug info.
! RUN: %flang_fc1 -emit-llvm -O2 -debug-info-kind=standalone \
! RUN:   -fdebug-info-for-profiling -fpseudo-probe-for-profiling -o - %s | FileCheck %s --check-prefix=PROBE-AND-DEBUG

! PROBE-PASS: Running pass: SampleProfileProbePass on {{.*}}

! PROBE-LABEL: define void @foo
! PROBE: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
! PROBE: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 2, i32 0, i64 -1)
! PROBE: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 4, i32 0, i64 -1)
! PROBE: call void @llvm.pseudoprobe(i64 [[#GUID]], i64 6, i32 0, i64 -1)
! PROBE: !llvm.pseudo_probe_desc = !{

! PROBE-AND-DEBUG: call void @llvm.pseudoprobe
! PROBE-AND-DEBUG: !llvm.pseudo_probe_desc = !{
! PROBE-AND-DEBUG: !DICompileUnit({{.*}}debugInfoForProfiling: true{{.*}})
! PROBE-AND-DEBUG: !DILexicalBlockFile({{.*}}discriminator:

subroutine foo(x)
   implicit none
   integer, intent(in) :: x
   if (x == 0) then
      call bar
   else
      call go
   end if
end subroutine foo
