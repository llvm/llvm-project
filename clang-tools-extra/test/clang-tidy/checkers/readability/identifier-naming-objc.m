// RUN: %check_clang_tidy %s readability-identifier-naming %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-identifier-naming.ObjcIvarPrefix: '_'}}' \
// RUN: --

@interface Foo {
    int _bar;
    int barWithoutPrefix;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for objc ivar 'barWithoutPrefix' [readability-identifier-naming]
    // CHECK-FIXES: int _barWithoutPrefix;
}
@end 
