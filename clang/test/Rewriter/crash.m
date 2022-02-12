// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 -o - %s
// rdar://5950938
@interface NSArray {}
+ (id)arrayWithObjects:(id)firstObj, ...;
@end

@interface NSConstantString {}
@end

int main(void) {
    id foo = [NSArray arrayWithObjects:@"1", @"2", @"3", @"4", @"5", @"6", @"7", @"8", @"9", @"10", @"11", @"12", 0];
    return 0;
}

// rdar://6291588
@protocol A
@end

@interface Foo
@end

void func(void) {
  id <A> obj = (id <A>)[Foo bar];
}

