! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
! CHECK: call foo( 0.)
! CHECK: call foo( 1.)
! CHECK: call foo( 2.)
! CHECK: call foo( 3.)
call foo( &
# 100 "bar.h"
         & 0.)
call foo( &
# 101 "bar.h"
         1.)
call foo( &
# 102 "bar.h"
         & 2. &
    & )
call foo( &
# 103 "bar.h"
         & 3. &
    )
end
