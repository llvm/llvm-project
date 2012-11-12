// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* Testcase for a problem where GCC allocated xqic to a register,
 * and did not have a VAR_DECL that explained the stack slot to LLVM.
 * Now the LLVM code synthesizes a stack slot if one is presented that
 * has not been previously recognized.  This is where alloca's named 
 * 'local' come from now. 
 */

typedef struct {
  short x;
} foostruct;

int foo(foostruct ic);

void test() {
  foostruct xqic;
  foo(xqic);
}


