// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// Do not issue error if 'ivar' used previously belongs to the inherited class
// and has same name as @dynalic property in current class.

typedef signed char BOOL;

@protocol IDEBuildable
@property (readonly) BOOL hasRecursiveDependencyCycle;
@end

@protocol IDEBuildableProduct <IDEBuildable>
@end

@interface IDEBuildableSupportMixIn 
@property (readonly) BOOL hasRecursiveDependencyCycle;
@end

@interface Xcode3TargetBuildable <IDEBuildable>
{
  IDEBuildableSupportMixIn *_buildableMixIn;
}
@end

@interface Xcode3TargetProduct : Xcode3TargetBuildable <IDEBuildableProduct>
@end

@implementation Xcode3TargetBuildable
- (BOOL)hasRecursiveDependencyCycle
{
    return [_buildableMixIn hasRecursiveDependencyCycle];
}
@end

@implementation Xcode3TargetProduct
@dynamic hasRecursiveDependencyCycle;
@end
