// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: FileCheck  -check-prefix CHECK-LP --input-file=%t-rw.cpp %s
// rdar://9846759

typedef void (^dispatch_block_t)(void);

extern int printf(const char*, ...);

extern "C" dispatch_block_t Block_copy(dispatch_block_t aBlock);

int main (int argc, char *argv[]) {

  dispatch_block_t innerBlock = ^{printf("argc = %d\n", argc); };
  id innerObject = 0;

  printf("innerBlock is %x\n", innerBlock);

  dispatch_block_t wrapperBlock = ^{
    printf("innerBlock is %x %x\n", innerBlock, innerObject);
  };

  wrapperBlock();

  dispatch_block_t copiedBlock = Block_copy(wrapperBlock);
  copiedBlock();

  return 0;
}
// CHECK-LP: _Block_object_assign((void*)&dst->innerBlock, (void*)src->innerBlock, 7
// CHECK-LP: _Block_object_dispose((void*)src->innerBlock, 7
// CHECK-LP: _Block_object_assign((void*)&dst->innerObject, (void*)src->innerObject, 3
// CHECK-LP: _Block_object_dispose((void*)src->innerObject, 3
