// RUN: %clang_cc1 -triple i686-windows -fdeclspec -fsyntax-only -verify %s

__declspec(dllimport) typedef int typedef1;
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}
typedef __declspec(dllimport) int typedef2;
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}
typedef int __declspec(dllimport) typedef3;
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}
typedef __declspec(dllimport) void (*FunTy)();
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}
enum __declspec(dllimport) E { Val };
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}
struct __declspec(dllimport) Record {};
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, and Objective-C interfaces}}

__declspec(dllimport)
__attribute__((__objc_root_class__))
@interface NSObject
@end

__declspec(dllimport)
@interface I : NSObject
- (void)method;
@end

@implementation I
- (void)method {
}
@end

