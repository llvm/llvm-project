// Test reading of PCH without original input files.

// Generate the PCH, removing the original file:
// RUN: echo 'struct S{char c; int i; }; void foo() {}' > %t.h
// RUN: echo 'template <typename T> void tf() { T::foo(); }' >> %t.h
// RUN: %clang_cc1 -x c++ -emit-pch -o %t.h.pch %t.h
// RUN: rm %t.h

// Check diagnostic with location in original source:
// RUN: not %clang_cc1 -include-pch %t.h.pch -emit-obj -o %t.o %s 2> %t.stderr
// RUN: grep 'could not find file' %t.stderr

// Oftentimes on Windows there are open handles, and deletion will fail.
// REQUIRES: can-remove-opened-file

void qq(S*) {}

#ifdef REDECL
float foo() {return 0f;}
#endif

#ifdef INSTANTIATION
void f() {
  tf<int>();
}
#endif
