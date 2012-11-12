// Test reading of PCH without original input files.

// Generate the PCH, removing the original file:
// RUN: echo 'struct S{char c; int i; }; void foo() {}' > %t.h
// RUN: echo 'template <typename T> void tf() { T::foo(); }' >> %t.h
// RUN: %clang_cc1 -x c++ -emit-pch -o %t.h.pch %t.h

// %t.h might be touched by scanners as a hot file on Windows,
// to fail to remove %.h with single run.
// RUN: rm %t.h || rm %t.h || rm %t.h

// Check diagnostic with location in original source:
// RUN: not %clang_cc1 -include-pch %t.h.pch -emit-obj -o %t.o %s 2> %t.stderr
// RUN: grep 'could not find file' %t.stderr

void qq(S*) {}

#ifdef REDECL
float foo() {return 0f;}
#endif

#ifdef INSTANTIATION
void f() {
  tf<int>();
}
#endif
