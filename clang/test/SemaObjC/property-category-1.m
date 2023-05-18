// RUN: %clang_cc1 -fsyntax-only -verify -disable-objc-default-synthesize-properties -Wno-objc-root-class %s

@interface Object
+ (id)new;
@end

@interface ReadOnly : Object
{
  int _object;
  int _Anotherobject;
}
@property(readonly) int object;
@property(readonly) int Anotherobject;
@end

@interface ReadOnly ()
@property(readwrite) int object;
@property(readwrite, setter = myAnotherobjectSetter:) int Anotherobject;
@end

@implementation ReadOnly
@synthesize object = _object;
@synthesize  Anotherobject = _Anotherobject;
- (void) myAnotherobjectSetter : (int)val {
    _Anotherobject = val;
}
- (int) Anotherobject { return _Anotherobject; }
@end

int main(int argc, char **argv) {
    ReadOnly *test = [ReadOnly new];
    test.object = 12345;
    test.Anotherobject = 200;
    return test.object - 12345 + test.Anotherobject - 200;
}

///

@interface I0
@property(readonly) int p0;	// expected-note {{property declared here}}
@end 

@interface I0 (Cat0)
@end 

@interface I0 (Cat1)
@end 
  
@implementation I0	// expected-warning {{property 'p0' requires method 'p0' to be define}}
- (void) foo {
  self.p0 = 0; // expected-error {{assignment to readonly property}}
}
@end
