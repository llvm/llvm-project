// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_global' %t  | count 7
// RUN: grep -F '@objc_assign_ivar' %t  | count 5
// RUN: grep -F '@objc_assign_strongCast' %t  | count 8
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_global' %t  | count 7
// RUN: grep -F '@objc_assign_ivar' %t  | count 5
// RUN: grep -F '@objc_assign_strongCast' %t  | count 8

extern id **somefunc(void);
extern id *somefunc2(void);


// Globals

id W, *X, **Y;

void func(id a, id *b, id **c) {
   static id w, *x, **y;
   W = a;  
   w = a;
   X = b;
   x = b; 
   Y = c;
   y = c; 
}

// Instances

@interface something {
    id w, *x, **y;
}
@end

@implementation something
- (void)amethod {
    id badIdea = *somefunc2();
    w = badIdea;
    x = &badIdea;
    y = &x;
}
@end

typedef struct {
    int junk;
    id  alfred;
} AStruct;

void funct2(AStruct *aptr) {
    id **ppptr = somefunc();
    aptr->alfred = 0;
    **ppptr = aptr->alfred;
    *ppptr = somefunc2(); 
}

typedef const struct __CFString * CFStringRef;
@interface DSATextSearch {
__strong CFStringRef *_documentNames;
  struct {
    id *innerNames;
    struct {
      id *nestedDeeperNames; 
      struct I {
         id *is1;
         id is2[5];
      } arrI [3];
    } inner_most;
  } inner;

}
- filter;
@end
@implementation DSATextSearch
- filter {
  int filteredPos = 0;
  _documentNames[filteredPos] = 0; // storing into an element of array ivar. objc_assign_strongCast is needed.
  inner.innerNames[filteredPos] = 0;
  inner.inner_most.nestedDeeperNames[filteredPos] = 0;
  inner.inner_most.arrI[3].is1[5] = 0;
  inner.inner_most.arrI[3].is2[5] = 0;
}
@end

