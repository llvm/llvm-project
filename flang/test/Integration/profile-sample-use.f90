! Test to check the working of option "-fprofile-sample-use".
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone -fprofile-sample-use=%S/inputs/pgo-sample.prof -o - %s | FileCheck %s

! CHECK: attributes #[[A:.*]] = { {{.*}}"use-sample-profile"{{.*}} }
! CHECK: !{i32 {{.*}}, !"ProfileSummary"{{.*}}}
! CHECK: !{!"ProfileFormat", !"SampleProfile"}
! CHECK: !{!"TotalCount", i64 100}
! CHECK: !{!"MaxCount", i64 100}
! CHECK: !{!"MaxInternalCount", i64 0}
! CHECK: !{!"MaxFunctionCount", i64 100}
! CHECK: !{!"NumCounts", i64 1}
! CHECK: !{!"NumFunctions", i64 1}
! CHECK: !{!"IsPartialProfile", i64 0}
! CHECK: !{!"PartialProfileRatio", double 0.000000e+00}
! CHECK: distinct !DISubprogram(name: "hot", linkageName: "hot_", scope: !1
! CHECK: !{!"function_entry_count", i64 101}

integer function hot(x)
   integer, intent(in) :: x
   hot = x * 2
end function hot

integer function cold(x)
   integer, intent(in) :: x
   cold = x - 10
end function cold

program test_sample_use
   implicit none
   integer :: i, r
   do i = 1, 100
      r = hot(i)
   end do
end program test_sample_use
