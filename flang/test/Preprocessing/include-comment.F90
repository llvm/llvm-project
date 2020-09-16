! RUN: %f18 -I%S -E %s 2>&1 | FileCheck %s
! CHECK-NOT: :3:
#include <empty.h> ! comment
! CHECK-NOT: :5:
#include <empty.h> /* comment */
! CHECK-NOT: :7:
#include <empty.h> !comment
! CHECK: :9:20: #include: extra stuff ignored after file name
#include <empty.h> comment
! CHECK-NOT: :11:
#include "empty.h" ! comment
! CHECK-NOT: :13:
#include "empty.h" /* comment */
! CHECK-NOT: :15:
#include "empty.h" !comment
! CHECK: :17:20: #include: extra stuff ignored after file name
#include "empty.h" comment
end
