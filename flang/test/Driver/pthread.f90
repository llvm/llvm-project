! In release 2.34, glibc removed libpthread as a separate library. All the
! pthread_* functions were subsumed into libc, so linking that is sufficient.
! However, when linking against older glibc builds, the explicit link of
! -pthread will be required. More details are here:
!
! https://developers.redhat.com/articles/2021/12/17/why-glibc-234-removed-libpthread#the_developer_view
!
! This makes it difficult to write a test that requires the -pthread flag in
! order to pass. Checking for the presence of -lpthread in the linker flags is
! not reliable since the linker could just skip the flag altogether if it is
! linking against a new libc implementation.

! RUN: %flang -### -pthread /dev/null -o /dev/null 2>&1 | FileCheck %s
! RUN: %flang -### -Xflang -pthread /dev/null -o /dev/null 2>&1 | FileCheck %s

! How the -pthread flag is handled is very platform-specific. A lot of that
! functionality is tested by clang, and the flag itself is handled by clang's
! driver that flang also uses. Instead of duplicating all that testing here,
! just check that the presence of the flag does not raise an error. If we need
! more customized handling of -pthread, the tests for that can be added here.
!
! CHECK-NOT: error:
