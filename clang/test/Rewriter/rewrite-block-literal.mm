// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

// rdar://11375908
typedef unsigned long size_t;

// rdar: // 11006566

void I( void (^)(void));
void (^noop)(void);

void nothing();
int printf(const char*, ...);

typedef void (^T) (void);

void takeblock(T);
int takeintint(int (^C)(int)) { return C(4); }

T somefunction() {
  if (^{ })
    nothing();

  noop = ^{};

  noop = ^{printf("\nClosure\n"); };

  I(^{ });

  return ^{printf("\nClosure\n"); };
}
void test2() {
  int x = 4;

  takeblock(^{ printf("%d\n", x); });

  while (1) {
    takeblock(^{ 
        while(1) break;  // ok
      });
    break;
  }
}

void test4() {
  void (^noop)(void) = ^{};
  void (*noop2)() = 0;
}

void myfunc(int (^block)(int)) {}

void myfunc3(const int *x);

void test5() {
  int a;

  myfunc(^(int abcd) {
      myfunc3(&a);
      return 1;
    });
}

void *X;

static int global_x = 10;
void (^global_block)(void) = ^{ printf("global x is %d\n", global_x); };

// CHECK: static __global_block_block_impl_0 __global_global_block_block_impl_0((void *)__global_block_block_func_0, &__global_block_block_desc_0_DATA);
// CHECK: void (*global_block)(void) = (void (*)())&__global_global_block_block_impl_0;

typedef void (^void_block_t)(void);

static const void_block_t myBlock = ^{ };

static const void_block_t myBlock2 = ^ void(void) { }; 
