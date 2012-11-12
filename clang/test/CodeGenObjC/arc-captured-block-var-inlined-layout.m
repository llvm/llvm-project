// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -O0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple i386-apple-darwin -O0 -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-i386 %s
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

//  Inline instruction for block variable layout: 0x0100
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 256 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 256 }
    void (^b)() = ^{
        x(bar);
    };    

// Inline instruction for block variable layout: 0x0210
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 528 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 528 }
    void (^c)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// Inline instruction for block variable layout: 0x0230
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 560 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 560 }
    void (^d)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// Inline instruction for block variable layout: 0x0231
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 561 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 561 }
    __weak id wid;
    id (^e)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// Inline instruction for block variable layout: 0x0235
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 565 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 565 }
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

// Inline instruction for block variable layout: 0x035
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 53 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 53 }
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

// Inline instruction for block variable layout: 0x01
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 1 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 1 }
    id (^h)() = ^{
        return wid;
    };

// Inline instruction for block variable layout: 0x020
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 32 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 32 }
    void (^ii)() = ^{
       byref_int = 1;
       byref_bab = 0;
    };

// Inline instruction for block variable layout: 0x0102
// CHECK: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i64 258 }
// CHECK-i386: i8* getelementptr inbounds ([6 x i8]* {{@.*}}, i32 0, i32 0), i32 258 }
    void (^jj)() = ^{
      x(bar);
      x(wid1);
      x(wid2);
    };
}
