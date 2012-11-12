// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
static int count;
static CGColorRef tmp = 0;

typedef struct S1  __attribute__ ((NSObject)) CGColorRef1; // expected-error {{__attribute ((NSObject)) is for pointer types only}}
typedef void *  __attribute__ ((NSObject)) CGColorRef2; // no-warning
typedef void * CFTypeRef;

@interface HandTested {
@public
    CGColorRef x;
}

@property(copy) CGColorRef x;
// rdar://problem/7809460
typedef struct CGColor * __attribute__((NSObject)) CGColorRefNoNSObject; // no-warning
@property (nonatomic, retain) CGColorRefNoNSObject color;
// rdar://problem/12197822
@property (strong) __attribute__((NSObject)) CFTypeRef myObj; // no-warning
@end

void setProperty(id self, id value)  {
  ((HandTested *)self)->x = value;
}

id getProperty(id self) {
     return (id)((HandTested *)self)->x;
}

@implementation HandTested
@synthesize x=x;
@synthesize myObj;
@dynamic color;
@end

int main(int argc, char *argv[]) {
    HandTested *to;
    to.x = tmp;  // setter
    if (tmp != to.x)
      to.x = tmp;
    return 0;
}

// rdar://10453342
@interface I
{
   __attribute__((NSObject)) void * color; // expected-warning {{__attribute ((NSObject)) may be put on a typedef only, attribute is ignored}}
}
  // <rdar://problem/10930507>
@property (nonatomic, retain) __attribute__((NSObject)) CGColorRefNoNSObject color; // // no-warning
@end
void test_10453342() {
    char* __attribute__((NSObject)) string2 = 0; // expected-warning {{__attribute ((NSObject)) may be put on a typedef only, attribute is ignored}}
}

// rdar://11569860
@interface A { int i; }
@property(retain) __attribute__((NSObject)) int i; // expected-error {{__attribute ((NSObject)) is for pointer types only}} \
  						   // expected-error {{property with 'retain (or strong)' attribute must be of object type}}
@end

@implementation A
@synthesize i;
@end

