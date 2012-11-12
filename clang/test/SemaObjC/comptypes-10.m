// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
//rdar: //8591619
// pr8453

@protocol NSCopying @end
@protocol NSPROTO @end
@protocol NSPROTO1 @end
@protocol NSPROTO2 @end

@interface NSObject <NSCopying, NSPROTO, NSPROTO1> {
    Class isa;
}
@end

void gorf(NSObject <NSCopying> *); // expected-note {{passing argument to parameter here}}

NSObject <NSCopying> *foo(id <NSCopying> bar, id id_obj)
{
 	NSObject <NSCopying> *Init = bar; // expected-warning {{initializing 'NSObject<NSCopying> *' with an expression of incompatible type 'id<NSCopying>'}}
        NSObject *Init1 = bar; // expected-warning {{initializing 'NSObject *' with an expression of incompatible type 'id<NSCopying>'}}

 	NSObject <NSCopying> *I = id_obj; 
        NSObject *I1 = id_obj; 
        gorf(bar);	// expected-warning {{passing 'id<NSCopying>' to parameter of incompatible type 'NSObject<NSCopying> *'}}

        gorf(id_obj);	

	return bar; 	// expected-warning {{returning 'id<NSCopying>' from a function with incompatible result type 'NSObject<NSCopying> *'}} 
}

void test(id <NSCopying, NSPROTO, NSPROTO2> bar)
{
  NSObject <NSCopying> *Init = bar; // expected-warning {{initializing 'NSObject<NSCopying> *' with an expression of incompatible type 'id<NSCopying,NSPROTO,NSPROTO2>'}}
}

// rdar://8843851
@interface NSObject (CAT)
+ (struct S*)Meth : (struct S*)arg;
@end

struct S {
 char *types;
};

@interface I
@end

@implementation I
- (struct S *)Meth : (struct S*)a {
  return [NSObject Meth : a];
}
@end
