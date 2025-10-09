! RUN: %flang -O1 -emit-llvm -S -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NOFASTMATH
! RUN: %flang -Ofast -emit-llvm -S -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-OFAST
! RUN: %flang -O1 -ffast-math -emit-llvm -S -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-FFAST-MATH

subroutine func
end subroutine func

! CHECK-NOFASTMATH-LABEL: define void @func_() local_unnamed_addr
! CHECK-NOFASTMATH-SAME: #[[ATTRS:[0-9]+]]
! CHECK-NOT: fp-math"=

! CHECK-OFAST-LABEL: define void @func_() local_unnamed_addr
! CHECK-OFAST-SAME: #[[ATTRS:[0-9]+]]
! CHECK-OFAST: attributes #[[ATTRS]] = { {{.*}}"no-infs-fp-math"="true" {{.*}}"no-nans-fp-math"="true" {{.*}}"no-signed-zeros-fp-math"="true" {{.*}}"unsafe-fp-math"="true"{{.*}} }

! CHECK-FFAST-MATH-LABEL: define void @func_() local_unnamed_addr
! CHECK-FFAST-MATH-SAME: #[[ATTRS:[0-9]+]]
! CHECK-FFAST-MATH: attributes #[[ATTRS]] = { {{.*}}"no-infs-fp-math"="true" {{.*}}"no-nans-fp-math"="true" {{.*}}"no-signed-zeros-fp-math"="true" {{.*}}"unsafe-fp-math"="true"{{.*}} }
