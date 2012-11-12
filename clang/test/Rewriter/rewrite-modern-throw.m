// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;

void *sel_registerName(const char *);

@interface Foo @end
void TRY();
void SPLATCH();
void MYTRY();
void MYCATCH();

void foo() {
  @try  { TRY(); } 
  @catch (...) { SPLATCH(); @throw; }
}

int main()
{

  @try  {
     MYTRY();
  }

  @catch (Foo* localException) {
     MYCATCH();
     @throw localException;
  }
  
  // no catch clause
  @try { } 
  @finally { }
}


@interface INST
{
  INST* throw_val;
}

- (id) ThrowThis;

- (void) MainMeth;

@end


@implementation INST
- (id) ThrowThis { return 0; }

- (void) MainMeth {
  @try  {
     MYTRY();
  }
  @catch (Foo* localException) {
     MYCATCH();
     @throw [self ThrowThis];
  }
  @catch (...) {
    @throw [throw_val ThrowThis];
  }
}
@end
