// UNSUPPORTED: system-windows

// Reset test
// rm -f %t-ref.o %t-readonly.o

// Create a reference file
// RUN: %clang -c %s -o %t-ref.o

// Compile something, mark the output as read-only and expect it to be replaced
// (permission bits of the file itself are irrelevant)
// RUN: touch %t-readonly.o
// RUN: chmod 000 %t-readonly.o
// RUN: %clang -c %s -o %t-readonly.o
// RUN: cmp %t-ref.o %t-readonly.o

void foo() {}
