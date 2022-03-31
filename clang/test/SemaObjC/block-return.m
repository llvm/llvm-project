// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -fobjc-gc-only %s
// expected-no-diagnostics
// rdar://8979379

@interface NSString
- (__attribute__((objc_gc(strong))) const char *)UTF8String;
@end

int main(void) {
__attribute__((objc_gc(strong))) char const *(^libraryNameForIndex)(void) = ^(void) {
        NSString *moduleName;
        return [moduleName UTF8String];
    };
}
