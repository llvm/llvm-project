// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s

@interface Foo {
  __unsafe_unretained id unsafe_ivar;
}

@property (assign,nonatomic) id unsafe_prop;

- (id)init;
+ (id)new;
+ (id)alloc;

-(void)Meth;
@end

@implementation Foo
@synthesize unsafe_prop;
-(id)init { return self; }
+(id)new { return 0; }
+(id)alloc { return 0; }

-(void)Meth {
  self.unsafe_prop = [Foo new]; // expected-warning {{assigning retained object to unsafe property}}
  self->unsafe_ivar = [Foo new]; // expected-warning {{assigning retained object to unsafe_unretained}}
  self.unsafe_prop = [[Foo alloc] init]; // expected-warning {{assigning retained object to unsafe property}}
  self->unsafe_ivar = [[Foo alloc] init]; // expected-warning {{assigning retained object to unsafe_unretained}}

  __unsafe_unretained id unsafe_var;
  unsafe_var = [Foo new]; // expected-warning {{assigning retained object to unsafe_unretained}}
  unsafe_var = [[Foo alloc] init]; // expected-warning {{assigning retained object to unsafe_unretained}}
}
@end

void bar(Foo *f) {
  f.unsafe_prop = [Foo new]; // expected-warning {{assigning retained object to unsafe property}}

  __unsafe_unretained id unsafe_var;
  unsafe_var = [Foo new]; // expected-warning {{assigning retained object to unsafe_unretained}}
  unsafe_var = [[Foo alloc] init]; // expected-warning {{assigning retained object to unsafe_unretained}}
}
