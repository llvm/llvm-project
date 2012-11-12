// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface Object
- (id)new;
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
@end

int main(int argc, char **argv) {
    ReadOnly *test = [ReadOnly new];
    test.object = 12345;
    test.Anotherobject = 200;
    return test.object - 12345 + test.Anotherobject - 200;
}

