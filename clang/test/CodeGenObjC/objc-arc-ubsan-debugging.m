// RUN: %clang -x objective-c -target arm64-apple-macos12.0 -fobjc-arc -std=gnu99  -O0 -fsanitize=undefined -fsanitize=nullability -c %s -v -g

@interface NSString
@end

struct A {
    NSString *a;
};

NSString* _Nonnull foo()
{
    struct A a;
    return 0;
}
