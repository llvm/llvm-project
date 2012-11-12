// Test reading of PCH with changed location of original input files,
// i.e. invoking header search.
// REQUIRES: shell

// Generate the original files:
// RUN: rm -rf %t_orig %t_moved
// RUN: mkdir -p %t_orig/sub %t_orig/sub2
// RUN: echo 'struct orig_sub{char c; int i; };' > %t_orig/sub/orig_sub.h
// RUN: echo 'void orig_sub2_1();' > %t_orig/sub2/orig_sub2_1.h
// RUN: echo '#include "orig_sub2_1.h"' > %t_orig/sub2/orig_sub2.h
// RUN: echo 'template <typename T> void tf() { orig_sub2_1(); T::foo(); }' >> %t_orig/sub2/orig_sub2.h
// RUN: echo 'void foo() {}' > %t_orig/tmp2.h
// RUN: echo '#include "tmp2.h"' > %t_orig/all.h
// RUN: echo '#include "sub/orig_sub.h"' >> %t_orig/all.h
// RUN: echo '#include "orig_sub2.h"' >> %t_orig/all.h
// RUN: echo 'int all();' >> %t_orig/all.h

// Generate the PCH:
// RUN: cd %t_orig && %clang_cc1 -x c++ -emit-pch -o all.h.pch -Isub2 all.h
// RUN: cp -pR %t_orig %t_moved

// Check diagnostic with location in original source:
// RUN: %clang_cc1 -include-pch all.h.pch -I%t_moved -I%t_moved/sub2 -Wpadded -emit-obj -o %t.o %s 2> %t.stderr
// RUN: grep 'struct orig_sub' %t.stderr

// Check diagnostic with 2nd location in original source:
// RUN: not %clang_cc1 -DREDECL -include-pch all.h.pch -I%t_moved -I%t_moved/sub2 -emit-obj -o %t.o %s 2> %t.stderr
// RUN: grep 'void foo' %t.stderr

// Check diagnostic with instantiation location in original source:
// RUN: not %clang_cc1 -DINSTANTIATION -include-pch all.h.pch -I%t_moved -I%t_moved/sub2 -emit-obj -o %t.o %s 2> %t.stderr
// RUN: grep 'orig_sub2_1' %t.stderr

void qq(orig_sub*) {all();}

#ifdef REDECL
float foo() {return 0;}
#endif

#ifdef INSTANTIATION
void f() {
  tf<int>();
}
#endif
