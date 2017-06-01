// RUN: %clang_cc1 -x objective-c -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -fcxx-exceptions -fexceptions  -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

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

// rdar://13186010
@class NSDictionary, NSException;
@class NSMutableDictionary;

@interface NSString
+ (id)stringWithFormat:(NSString *)format, ... ;
@end

@interface  NSException
+ (NSException *)exceptionWithName:(NSString *)name reason:(NSString *)reason userInfo:(NSDictionary *)userInfo;
@end
id *_imp__NSInvalidArgumentException;

@interface NSSetExpression @end

@implementation NSSetExpression
-(id)expressionValueWithObject:(id)object context:(NSMutableDictionary*)bindings {
    id leftSet;
    id rightSet;
    @throw [NSException exceptionWithName: *_imp__NSInvalidArgumentException reason: [NSString stringWithFormat: @"Can't evaluate set expression; left subexpression not a set (lhs = %@ rhs = %@)", leftSet, rightSet] userInfo: 0];

    return leftSet ;
}
@end

