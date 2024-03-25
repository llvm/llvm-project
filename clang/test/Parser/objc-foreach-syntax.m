// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

struct __objcFastEnumerationState; 
@implementation MyList // expected-warning {{cannot find interface declaration for 'MyList'}}
- (unsigned int)countByEnumeratingWithState:  (struct __objcFastEnumerationState *)state objects:  (id *)items count:(unsigned int)stackcount
{
     return 0;
}
@end


int LOOP(void);

@implementation MyList (BasicTest) 
- (void)compilerTestAgainst {
MyList * el; 
     for (el in @"foo") 
	  { LOOP(); }
}
@end


static int test7(id keys) {
  for (id key; in keys) ;  // expected-error {{use of undeclared identifier 'in'}} \
                           // expected-error {{expected ';' in 'for' statement specifier}}
}
