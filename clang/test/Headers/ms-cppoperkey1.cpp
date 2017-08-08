// RUN: %clang_cc1 \
// RUN:     -fms-compatibility -x c++-cpp-output \
// RUN:     -ffreestanding -fsyntax-only -Werror \
// RUN:     %s -verify


# 1 "t.cpp"
# 1 "query.h" 1 3 4
// MS header <query.h> uses operator keyword as field name.  
// Compile without syntax errors.
struct tagRESTRICTION
  {
   union _URes 
     {
       int or; // Note use of cpp operator token
     } res;
  };
   ;

int aa ( int x)
{
  // In system header code, treat operator keyword as identifier.
  if ( // expected-note{{to match this '('}}
    x>1 or x<0) return 1; // expected-error{{expected ')'}}
  else return 0;  
}

