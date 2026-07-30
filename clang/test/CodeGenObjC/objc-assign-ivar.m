// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_ivar' %t  | count 14

typedef struct {
  id  element;
  id elementArray[10];
  __strong id cfElement;
  __strong id cfElementArray[10];
} struct_with_ids_t;


@interface NSString @end

@interface Foo  {
@public
// assignments to any/all of these fields should generate objc_assign_ivar
  __strong id dict;
  __strong id dictArray[3];
  id ivar;
  id array[10];
  id nsobject;
  NSString *stringArray[10];
  struct_with_ids_t inner;

  Foo *obj[20];
  short idx[5];
}
@end

// The test cases
int IvarAssigns;
void *rhs = 0;
#define ASSIGNTEST(expr, global) expr = rhs

void testIvars(void) {
  Foo *foo;
  ASSIGNTEST(foo->ivar, IvarAssigns);                                   // objc_assign_ivar
  ASSIGNTEST(foo->dict, IvarAssigns);                                   // objc_assign_ivar
  ASSIGNTEST(foo->dictArray[0], IvarAssigns);                           // objc_assign_ivar
  ASSIGNTEST(foo->array[0], IvarAssigns);                               // objc_assign_ivar
  ASSIGNTEST(foo->nsobject, IvarAssigns);                               // objc_assign_ivar
  ASSIGNTEST(foo->stringArray[0], IvarAssigns);                         // objc_assign_ivar
  ASSIGNTEST(foo->inner.element, IvarAssigns);                          // objc_assign_ivar
  ASSIGNTEST(foo->inner.elementArray[0], IvarAssigns);                  // objc_assign_ivar
  ASSIGNTEST(foo->inner.cfElement, IvarAssigns);                        // objc_assign_ivar
  ASSIGNTEST(foo->inner.cfElementArray[0], IvarAssigns);                // objc_assign_ivar
  int counter=1;
  ASSIGNTEST(foo->obj[5], IvarAssigns);                 // objc_assign_ivar
  ASSIGNTEST(foo->obj[++counter], IvarAssigns);         // objc_assign_ivar
  foo->idx[++counter] = 15;
  ASSIGNTEST(foo->obj[foo->idx[2]], IvarAssigns);       // objc_assign_ivar
}
