@class AClass;

@protocol Protocol

- (void)methodAttribute __attribute__((availability(macos, introduced=10.10)));

- (void)parameterTypeAttribute:(AClass * _Nullable)c;

- (void)parameterTypeAttribute2:(AClass * _Nonnull)c;

- (void)parameterAttribute:(int)p __attribute__((annotate("test")));

@end

@interface Base
@end

@interface I1 : Base<Protocol>

@end
// CHECK1: "- (void)methodAttribute;\n\n- (void)parameterAttribute:(int)p;\n\n- (void)parameterTypeAttribute2:(AClass * _Nonnull)c;\n\n- (void)parameterTypeAttribute:(AClass * _Nullable)c;\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1

@implementation I1

@end
// CHECK1: "- (void)methodAttribute { \n  <#code#>\n}\n\n- (void)parameterAttribute:(int)p { \n  <#code#>\n}\n\n- (void)parameterTypeAttribute2:(AClass * _Nonnull)c { \n  <#code#>\n}\n\n- (void)parameterTypeAttribute:(AClass * _Nullable)c { \n  <#code#>\n}\n\n" [[@LINE-1]]:1 -> [[@LINE-1]]:1
// RUN: clang-refactor-test perform -action fill-in-missing-protocol-stubs -at=%s:18:1 -at=%s:23:1 %s | FileCheck --check-prefix=CHECK1 %s

