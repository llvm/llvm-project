// Check UNIX conformance for cc/c89/c99
// When c99 encounters a compilation error that causes an object file not to be
// created, it shall write a diagnostic to standard error and continue to
// compile other source code operands, but it shall not perform the link phase
// and it shall return a non-zero exit status.

// When given multiple .c files to compile, clang compiles them in order until
// it hits an error, at which point it stops.
//
// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: cd %t-dir
//
// RUN: touch %t-dir/1.c
// RUN: echo "invalid C code" > %t-dir/2.c
// RUN: touch %t-dir/3.c
// RUN: echo "invalid C code" > %t-dir/4.c
// RUN: touch %t-dir/5.c
// RUN: not %clang -S %t-dir/1.c %t-dir/2.c %t-dir/3.c %t-dir/4.c %t-dir/5.c
// RUN: test -f %t-dir/1.s
// RUN: test ! -f %t-dir/2.s
// RUN: test -f %t-dir/3.s
// RUN: test ! -f %t-dir/4.s
// RUN: test -f %t-dir/5.s
