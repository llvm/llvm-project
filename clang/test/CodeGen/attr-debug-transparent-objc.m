// RUN: %clang -gdwarf -emit-llvm -S %s -o - | FileCheck %s


@interface ObjCClass
- (void)foo __attribute__((debug_transparent));
@end

@implementation ObjCClass
- (void)foo {}
@end


// CHECK: DISubprogram(name: "-[ObjCClass foo]"{{.*}} DISPFlagIsDebugTransparent
