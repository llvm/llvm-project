// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

typedef struct CGColor * __attribute__ ((NSObject)) CGColorRef;
typedef struct CGColor * __attribute__((NSObject(12))) Illegal;  // expected-error {{'NSObject' attribute takes no arguments}}

static int count;
static CGColorRef tmp = 0;

typedef struct S1  __attribute__ ((NSObject)) CGColorRef1; // expected-error {{'NSObject' attribute is for pointer types only}}
typedef void *  __attribute__ ((NSObject)) CGColorRef2; // no-warning
typedef void * CFTypeRef;

@interface HandTested {
@public
    CGColorRef x;
}

@property(copy) CGColorRef x;
typedef struct CGColor * __attribute__((NSObject)) CGColorRefNoNSObject; // no-warning
@property (nonatomic, retain) CGColorRefNoNSObject color;
@property (strong) __attribute__((NSObject)) CFTypeRef myObj; // no-warning
@property (strong, nullable) CGColorRefNoNSObject color2; // no-warning
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

@interface I
{
   __attribute__((NSObject)) void * color; // expected-warning {{'NSObject' attribute may be put on a typedef only; attribute is ignored}}
}
@property (nonatomic, retain) __attribute__((NSObject)) CGColorRefNoNSObject color; // // no-warning
@end
void test_10453342(void) {
    char* __attribute__((NSObject)) string2 = 0; // expected-warning {{'NSObject' attribute may be put on a typedef only; attribute is ignored}}
}

@interface A { int i; }
@property(retain) __attribute__((NSObject)) int i; // expected-error {{'NSObject' attribute is for pointer types only}} \
  						   // expected-error {{property with 'retain (or strong)' attribute must be of object type}}
@end

@implementation A
@synthesize i;
@end

