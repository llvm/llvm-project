// RUN: %clang_cc1 -triple i686-windows -fdeclspec -fsyntax-only -verify %s

__declspec(dllexport) typedef int typedef1;
// expected-warning@-1{{'dllexport' attribute only applies to functions, variables, classes, and Objective-C interfaces}}
typedef __declspec(dllexport) int typedef2;
// expected-warning@-1{{'dllexport' attribute only applies to}}
typedef int __declspec(dllexport) typedef3;
// expected-warning@-1{{'dllexport' attribute only applies to}}
typedef __declspec(dllexport) void (*FunTy)(void);
// expected-warning@-1{{'dllexport' attribute only applies to}}
enum __declspec(dllexport) E { Val };
// expected-warning@-1{{'dllexport' attribute only applies to}}
struct __declspec(dllexport) Record {};
// expected-warning@-1{{'dllexport' attribute only applies to}}

__declspec(dllexport)
__attribute__((__objc_root_class__))
@interface NSObject
@end

__declspec(dllexport)
@interface I : NSObject
- (void)method;
@end

@implementation I
- (void)method {
}
@end

