// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin9 %s -emit-llvm -o - | FileCheck %s -check-prefix=X64
// RUN: %clang_cc1 -fblocks -triple i686-apple-darwin9 %s -emit-llvm -o - | FileCheck %s -check-prefix=X32

// X64: @.str = private unnamed_addr constant [6 x i8] c"v8@?0\00" 
// X64: @__block_literal_global = internal constant {{.*}} { i8** @_NSConcreteGlobalBlock, i32 1342177280,
// X64: @.str1 = private unnamed_addr constant [12 x i8] c"i16@?0c8f12\00"
// X64:   store i32 1073741824, i32*

// X32: [[STR1:@.*]] = private unnamed_addr constant [6 x i8] c"v4@?0\00" 
// X32: @__block_descriptor_tmp = internal constant [[FULL_DESCRIPTOR_T:.*]] { i32 0, i32 20, i8* getelementptr inbounds ([6 x i8]* [[STR1]], i32 0, i32 0), i8* null }
// X32: @__block_literal_global = internal constant [[GLOBAL_LITERAL_T:.*]] { i8** @_NSConcreteGlobalBlock, i32 1342177280, i32 0, i8* bitcast (void (i8*)* @global_block_invoke{{.*}} to i8*), [[DESCRIPTOR_T:%.*]]* bitcast ([[FULL_DESCRIPTOR_T]]* @__block_descriptor_tmp to {{%.*}}*) }
// X32: [[STR2:@.*]] = private unnamed_addr constant [11 x i8] c"i12@?0c4f8\00"
// X32: @__block_descriptor_tmp{{.*}} = internal constant [[FULL_DESCRIPTOR_T]] { i32 0, i32 24, i8* getelementptr inbounds ([11 x i8]* [[STR2]], i32 0, i32 0), i8* null }
// X32:   store i32 1073741824, i32*

// rdar://7635294


int globalInt;
void (^global)(void) = ^{ ++globalInt; };

    
void foo(int param) {
   extern int rand(void);
   extern void rand_r(int (^b)(char x, float y));   // name a function present at runtime
   while (param--)
      rand_r(^(char x, float y){ return x + (int)y + param + rand(); });  // generate a local block binding param
}

#if 0
#include <stdio.h>
enum {
    BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
    BLOCK_HAS_CXX_OBJ =       (1 << 26),
    BLOCK_IS_GLOBAL =         (1 << 28),
    BLOCK_HAS_DESCRIPTOR =    (1 << 29),
    BLOCK_HAS_OBJC_TYPE  =    (1 << 30)
};

struct block_descriptor_big {
    unsigned long int reserved;
    unsigned long int size;
    void (*copy)(void *dst, void *src); // conditional on BLOCK_HAS_COPY_DISPOSE
    void (*dispose)(void *);            // conditional on BLOCK_HAS_COPY_DISPOSE
    const char *signature;                  // conditional on BLOCK_HAS_OBJC
    const char *layout;                 // conditional on BLOCK_HAS_OBJC
};
struct block_descriptor_small {
    unsigned long int reserved;
    unsigned long int size;
    const char *signature;              // conditional on BLOCK_HAS_OBJC
    const char *layout;                 // conditional on BLOCK_HAS_OBJC
};

struct block_layout_abi { // can't change
  void *isa;
  int flags;
  int reserved; 
  void (*invoke)(void *, ...);
  struct block_descriptor_big *descriptor;
};

const char *getBlockSignature(void *block) {
   struct block_layout_abi *layout = (struct block_layout_abi *)block;
   if ((layout->flags & BLOCK_HAS_OBJC_TYPE) != BLOCK_HAS_OBJC_TYPE) return NULL;
   if (layout->flags & BLOCK_HAS_COPY_DISPOSE) 
      return layout->descriptor->signature;
   else
      return ((struct block_descriptor_small *)layout->descriptor)->signature;
}
  
    
   
int main(int argc, char *argv[]) {
   printf("desired global flags: %d\n", BLOCK_IS_GLOBAL  | BLOCK_HAS_OBJC_TYPE);
   printf("desired stack flags: %d\n",  BLOCK_HAS_OBJC_TYPE);
   
   printf("types for global: %s\n", getBlockSignature(global));
   printf("types for local: %s\n", getBlockSignature(^int(char x, float y) { return (int)(y + x); }));
   return 0;
}

/*
x86_64
desired global flags: 1342177280
desired stack flags: 1073741824
types for global: v8@?0
types for local: i16@?0c8f12

i386
desired global flags: 1342177280
desired stack flags: 1073741824
types for global: v4@?0
types for local: i12@?0c4f8
*/
#endif
