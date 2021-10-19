// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null %s > %t-64.layout
// RUN: FileCheck --input-file=%t-64.layout %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple i386-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null  %s > %t-32.layout
// RUN: FileCheck --input-file=%t-32.layout %s
// rdar://12184410

void x(id y) {}
void y(int a) {}

extern id opaque_id();

void f() {
    __block int byref_int = 0;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong id strong_void_sta;
    __block id byref_bab = (id)0;
    __block id bl_var1;

// CHECK: Inline block variable layout: 0x0100, BL_STRONG:1, BL_OPERATOR:0
    void (^b)() = ^{
        x(bar);
    };    

// CHECK: Inline block variable layout: 0x0210, BL_STRONG:2, BL_BYREF:1, BL_OPERATOR:0
    void (^c)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// CHECK: Inline block variable layout: 0x0230, BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
    void (^d)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// CHECK: Inline block variable layout: 0x0231, BL_STRONG:2, BL_BYREF:3, BL_WEAK:1, BL_OPERATOR:0
    __weak id wid;
    id (^e)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// CHECK: Inline block variable layout: 0x0235, BL_STRONG:2, BL_BYREF:3, BL_WEAK:5, BL_OPERATOR:0
    __weak id wid1, wid2, wid3, wid4;
    id (^f)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        x(wid1);
        x(wid2);
        x(wid3);
        x(wid4);
        return wid;
    };

// CHECK: Inline block variable layout: 0x035, BL_BYREF:3, BL_WEAK:5, BL_OPERATOR:0
    id (^g)() = ^{
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        x(wid1);
        x(wid2);
        x(wid3);
        x(wid4);
        return wid;
    };

// CHECK: Inline block variable layout: 0x01, BL_WEAK:1, BL_OPERATOR:0
    id (^h)() = ^{
        return wid;
    };

// CHECK: Inline block variable layout: 0x020, BL_BYREF:2, BL_OPERATOR:0
    void (^ii)() = ^{
       byref_int = 1;
       byref_bab = 0;
    };

// CHECK: Inline block variable layout: 0x0102, BL_STRONG:1, BL_WEAK:2, BL_OPERATOR:0
    void (^jj)() = ^{
      x(bar);
      x(wid1);
      x(wid2);
    };
}

// rdar://12752901
@class NSString;
extern void NSLog(NSString *format, ...);
typedef void (^dispatch_block_t)(void);
int main() {
        __strong NSString *s1 = 0;
        __strong NSString *s2 = 0;
        __weak NSString *w1 = 0;


// CHECK: Inline block variable layout: 0x0201, BL_STRONG:2, BL_WEAK:1, BL_OPERATOR:0
        dispatch_block_t block2 = ^{
                NSLog(@"%@, %@, %@", s1, w1, s2);
        };
}
