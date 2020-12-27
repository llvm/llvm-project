// Check that we can operate on files from /dev/fd.
// REQUIRES: dev-fd-fs
// REQUIRES: shell

// Check reading from named pipes. We cat the input here instead of redirecting
// it to ensure that /dev/fd/0 is a named pipe, not just a redirected file.
//
// RUN: cat %s | %clang -x c /dev/fd/0 -E > %t
// RUN: FileCheck --check-prefix DEV-FD-INPUT < %t %s
//
// RUN: cat %s | %clang -x c %s -E -DINCLUDE_FROM_STDIN > %t
// RUN: FileCheck --check-prefix DEV-FD-INPUT \
// RUN:           --check-prefix DEV-FD-INPUT-INCLUDE < %t %s
//
// DEV-FD-INPUT-INCLUDE: int w;
// DEV-FD-INPUT: int x;
// DEV-FD-INPUT-INCLUDE: int y;


// Check writing to /dev/fd named pipes. We use cat here as before to ensure we
// get a named pipe.
//
// RUN: %clang -x c %s -E -o /dev/fd/1 | cat > %t
// RUN: FileCheck --check-prefix DEV-FD-FIFO-OUTPUT < %t %s
//
// DEV-FD-FIFO-OUTPUT: int x;


// Check writing to /dev/fd regular files.
//
// RUN: %clang -x c %s -E -o /dev/fd/1 > %t
// RUN: FileCheck --check-prefix DEV-FD-REG-OUTPUT < %t %s
//
// DEV-FD-REG-OUTPUT: int x;

#ifdef INCLUDE_FROM_STDIN
#undef INCLUDE_FROM_STDIN
int w;
#include "/dev/fd/0"
int y;
#else
int x;
#endif
