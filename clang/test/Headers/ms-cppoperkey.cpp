// RUN: %clang_cc1 \
// RUN:     -fms-compatibility -x c++-cpp-output \
// RUN:     -ffreestanding -fsyntax-only -Werror \
// RUN:     %s -verify
// expected-no-diagnostics
# 1 "t.cpp"
# 1 "query.h" 1 3
// MS header <query.h> uses operator keyword as field name.  
// Compile without syntax errors.
struct tagRESTRICTION
  {
   union _URes 
     {
       int or; // Note use of cpp operator token
     } res;
  };
