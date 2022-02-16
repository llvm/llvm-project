// RUN: %clang_cc1 -fblocks -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null %s > %t-64.layout
// RUN: FileCheck --input-file=%t-64.layout %s
// RUN: %clang_cc1 -fblocks -fobjc-runtime-has-weak -fobjc-arc -triple i386-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null %s > %t-32.layout
// RUN: FileCheck --input-file=%t-32.layout %s
// rdar://12184410
// rdar://12184410

void x(id y) {}
void y(int a) {}

extern id opaque_id(void);
__weak id wid;

void f(void) {
    __block int byref_int = 0;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong id strong_void_sta;
    __block id byref_bab = (id)0;
    __block id bl_var1;

// block variable layout: BL_STRONG:1, BL_OPERATOR:0
// CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
    void (^b)(void) = ^{
        x(bar);
    };    

// block variable layout: BL_STRONG:2, BL_BYREF:1, BL_OPERATOR:0
// CHECK: Inline block variable layout: 0x0210, BL_STRONG:2, BL_BYREF:1, BL_OPERATOR:0
    void (^c)(void) = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// block variable layout: BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
// CHECK: Inline block variable layout: 0x0230, BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
    void (^d)(void) = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// block variable layout: BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
// CHECK: Inline block variable layout: 0x0230, BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
    id (^e)(void) = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// CHECK: Inline block variable layout: 0x020, BL_BYREF:2, BL_OPERATOR:0
    void (^ii)(void) = ^{
       byref_int = 1;
       byref_bab = 0;
    };
}
