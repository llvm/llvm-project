// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp -emit-llvm -o %t-rw.ll
// RUN: FileCheck --input-file=%t-rw.ll %s

typedef void (^void_block_t)(void);

static const void_block_t myblock = ^{
        
};

// CHECK: myblock = internal global
