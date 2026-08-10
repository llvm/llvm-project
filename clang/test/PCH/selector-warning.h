typedef struct objc_selector    *SEL;

@interface Foo 
- (void) NotOK;
@end

@implementation Foo
- (void) foo
{
  SEL a = @selector(b1ar); 
  a = @selector(b1ar); 
  a = @selector(bar);
  a = @selector(ok);	// expected-warning {{unimplemented selector 'ok'}}
  a = @selector(ok);
  a = @selector(NotOK);	// expected-warning {{unimplemented selector 'NotOK'}}
  a = @selector(NotOK);

  a = @selector(clNotOk);	// expected-warning {{unimplemented selector 'clNotOk'}}

  a = @selector (cl1);
  a = @selector (cl2);
  a = @selector (instNotOk);	// expected-warning {{unimplemented selector 'instNotOk'}}
}
@end
