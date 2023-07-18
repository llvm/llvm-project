// RUN: %clang_cc1 -fblocks -emit-llvm %s -fobjc-gc -o - | FileCheck %s

// CHECK: objc_assign_strongCast
// rdar://5541393

typedef __SIZE_TYPE__ size_t;
void * malloc(size_t size);

typedef struct {
    void (^ivarBlock)(void);
} StructWithBlock_t;

int main(int argc, char *argv[]) {
   StructWithBlock_t *swbp = (StructWithBlock_t *)malloc(sizeof(StructWithBlock_t*));
   __block   int i = 10;
   // assigning a Block into an struct slot should elicit a write-barrier under GC
   swbp->ivarBlock = ^ { ++i; };
   return 0;
}
