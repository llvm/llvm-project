// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -print-ivar-layout -emit-llvm -o /dev/null %s > %t-64.layout
// RUN: FileCheck -check-prefix CHECK-LP64 --input-file=%t-64.layout %s
// rdar://12752901

struct S {
    int i1;
    id o1;
    struct V {
     int i2;
     id o2;
    } v1;
    int i3;
    id o3;
};

__weak id wid;
void x(id y) {}
void y(int a) {}

extern id opaque_id(void);

void f(void) {
    __block int byref_int = 0;
    char ch = 'a';
    char ch1 = 'b';
    char ch2 = 'c';
    short sh = 2;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong void *strong_void_sta;
    __block id byref_bab = (id)0;
    __block void *bl_var1;
    int i; double dob;

// The patterns here are a sequence of bytes, each saying first how
// many sizeof(void*) chunks to skip (high nibble) and then how many
// to scan (low nibble).  A zero byte says that we've reached the end
// of the pattern.
//
// All of these patterns start with 01 3x because the block header on
// LP64 consists of an isa pointer (which we're supposed to scan for
// some reason) followed by three words (2 ints, a function pointer,
// and a descriptor pointer).

// FIXME: do these really have to be named L_OBJC_CLASS_NAME_xxx?
// FIXME: sequences should never end in x0 00 instead of just 00

// Test 1
// byref int, short, char, char, char, id, id, strong void*, byref id
// CHECK-LP64: block variable layout for block: 0x01, 0x35, 0x10, 0x00
    void (^b)(void) = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(byref_bab);
    };    
    b();

// Test 2
// byref int, short, char, char, char, id, id, strong void*, byref void*, byref id
// 01 36 10 00
// CHECK-LP64: block variable layout for block: 0x01, 0x36, 0x10, 0x00
    void (^c)(void) = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(wid);
        bl_var1 = 0;
        x(byref_bab);
    };    
    c();

// Test 3
// byref int, short, char, char, char, id, id, byref void*, int, double, byref id
// 01 34 11 30 00
// FIXME: we'd get a better format here if we sorted by scannability, not just alignment
// CHECK-LP64: block variable layout for block: 0x01, 0x35, 0x30, 0x00
    void (^d)(void) = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x(wid);
        bl_var1 = 0; 
        y(i + dob);
        x(byref_bab);
    };    
    d();

// Test 4
// struct S (int, id, int, id, int, id)
// 01 41 11 11 00
// CHECK-LP64: block variable layout for block: 0x01, 0x41, 0x11, 0x11, 0x00
    struct S s2;
    void (^e)(void) = ^{
        x(s2.o1);
    };    
    e();
}

// Test 5 (unions/structs and their nesting):
void Test5(void) {
  struct S5 {
    int i1;
    id o1;
    struct V {
     int i2;
     id o2;
    } v1;
    int i3;
    union UI {
        void * i1;
        id o1;
        int i3;
        id o3;
    }ui;
  };

  union U {
        void * i1;
        id o1;
        int i3;
        id o3;
  }ui;

  struct S5 s2;
  union U u2;

// struct s2 (int, id, int, id, int, id?), union u2 (id?)
// 01 41 11 12 00
// CHECK-LP64: block variable layout for block: 0x01, 0x41, 0x11, 0x12, 0x00
  void (^c)(void) = ^{
    x(s2.ui.o1);
    x(u2.o1);
  };
  c();
}

// rdar: //8417746
void CFRelease(id);
void notifyBlock(id dependentBlock) {
 id singleObservationToken;
 id token;
 void (^b)(void);

// id, id, void(^)(void)
// 01 33 00
// CHECK-LP64: block variable layout for block: 0x01, 0x33, 0x00
 void (^wrapperBlock)(void) = ^(void) {
     CFRelease(singleObservationToken);
     CFRelease(singleObservationToken);
     CFRelease(token);
     CFRelease(singleObservationToken);
     b();
    };
 wrapperBlock();
}

void test_empty_block(void) {
// 01 00
// CHECK-LP64: block variable layout for block: 0x01, 0x30, 0x00
  void (^wrapperBlock)(void) = ^(void) {
  };
 wrapperBlock();
}

// rdar://16111839
typedef union { char ch[8];  } SS;
typedef struct { SS s[4]; } CS;
void test_union_in_layout(void) {
  CS cs;
  ^{ cs; };
}
