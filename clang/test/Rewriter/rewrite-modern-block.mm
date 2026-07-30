// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp

typedef unsigned long size_t;
typedef struct {
    char byte0;
    char byte1;
} CFUUIDBytes;

void x(void *);

void y() {
    __block CFUUIDBytes bytes;
    
    void (^bar)() = ^{
        x(&bytes);
    };
}

int foo() {
    __block int hello;
    return hello;
}

// rewriting multiple __block decls on wintin same decl stmt.
void radar7547630() {
  __block int BI1, BI2;

  __block float FLOAT1, FT2, FFFFFFFF3,
   FFFXXX4;

  __block void (^B)(), (^BB)();
}

// rewriting multiple __block decls on wintin same decl stmt
// with initializers.
int  rdar7547630(const char *keybuf, const char *valuebuf) {
  __block int BI1 = 1, BI2 = 2;

  double __block BYREFVAR = 1.34, BYREFVAR_NO_INIT, BYREFVAR2 = 1.37;

  __block const char *keys = keybuf, *values = valuebuf, *novalues;

  return BI2;
}

typedef struct _z {
    int location;
    int length;
} z;

z w(int loc, int len);

@interface rdar11326988
@end
@implementation rdar11326988 
- (void)y:(int)options {
    __attribute__((__blocks__(byref))) z firstRange = w(1, 0);
    options &= ~(1 | 2);
}
@end

int Test18799145() { return ^(){return 0;}(); }
