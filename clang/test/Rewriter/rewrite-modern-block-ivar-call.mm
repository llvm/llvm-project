// RUN: %clang_cc1 -E %s -o %t.m
// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %t.m -o %t-rw.cpp 
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

@interface Foo {
    void (^_block)(void);
}
@end

@implementation Foo
- (void)bar {
    _block();
}
@end

// CHECK: ((void (*)(struct __block_impl *))((struct __block_impl *)(*(void (**)(void))((char *)self + OBJC_IVAR_$_Foo$_block)))->FuncPtr)((struct __block_impl *)(*(void (**)(void))((char *)self + OBJC_IVAR_$_Foo$_block)));
