// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface Super @end

@interface NSArray : Super @end
@interface NSSet : Super @end

@protocol MyProtocol
- (void)myMethod;
@end

@protocol MyProtocol2 <MyProtocol>
- (void)myMethod2;
@end

@interface NSArray() <MyProtocol2>
@end

@interface NSSet() <MyProtocol>
@end

int main (int argc, const char * argv[]) {
    NSArray *array = (void*)0;
    NSSet *set = (void*)0;
    id <MyProtocol> instance = (argc) ? array : set;
    instance = (void*)0;
    return 0;
}

