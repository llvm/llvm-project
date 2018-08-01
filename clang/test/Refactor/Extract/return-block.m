// RUN: clang-refactor-test perform -action extract -selected=%s:16:27-18:4 %s -fobjc-arc | FileCheck %s
// RUN: clang-refactor-test perform -action extract -selected=%s:16:27-18:4 %s | FileCheck --check-prefix=NOARC %s
@interface I

@end

@implementation I

- (void) doStuff: (int)x block: (void (^)(int))block {
  
}

- (void)foo {}

- (void)viewDidLoad {
  [self doStuff: 2 block: ^(int returnCode) {
    [self foo];
  }];
}
// CHECK: "static void (^extracted(I *object))(int) {\nreturn ^(int returnCode) {\n    [object foo];\n  };\n}\n\n" [[@LINE-5]]:1 -> [[@LINE-5]]:1
// CHECK: "extracted(self)" [[@LINE-5]]:27 -> [[@LINE-3]]:4
// NOARC: "static void (^extracted(I *object))(int) {\nreturn [(^(int returnCode) {\n    [object foo];\n  }) copy];\n}\n\n" [[@LINE-7]]:1 -> [[@LINE-7]]:1

@end
