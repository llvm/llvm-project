! RUN: %flang -O1 -finstrument-functions -emit-llvm -S -o - %s 2>&1| FileCheck %s

subroutine func
end subroutine func

! CHECK: define void @func_()
! CHECK: {{.*}}call void @__cyg_profile_func_enter(ptr {{.*}}@func_, ptr {{.*}})
! CHECK: {{.*}}call void @__cyg_profile_func_exit(ptr {{.*}}@func_, ptr {{.*}})
! CHECK-NEXT: ret {{.*}}
