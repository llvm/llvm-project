// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -F%t/Frameworks %t/test.m -Wno-objc-property-implementation -Wno-incomplete-implementation \
// RUN:            -fmodules -fmodule-name=Target -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// RUN: %clang_cc1 -fsyntax-only -F%t/Frameworks -x objective-c++ %t/test.m -Wno-objc-property-implementation -Wno-incomplete-implementation \
// RUN:            -fmodules -fmodule-name=Target -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test anonymous TagDecl inside Objective-C interfaces are merged and ivars with these anonymous types are merged too.

//--- Frameworks/Foundation.framework/Headers/Foundation.h
@interface NSObject
@end

//--- Frameworks/Foundation.framework/Modules/module.modulemap
framework module Foundation {
  header "Foundation.h"
  export *
}

//--- Frameworks/Target.framework/Headers/Target.h
#import <Foundation/Foundation.h>
@interface TestClass : NSObject {
@public
  struct {
    struct { int x; int y; } left;
    struct { int w; int z; } right;
  } structIvar;
  union { int x; float y; } *unionIvar;
  enum { kX = 0, } enumIvar;
}
@property struct { int u; } prop;
#ifndef __cplusplus
- (struct { int v; })method;
#endif
@end

@interface TestClass() {
@public
  struct { int y; } extensionIvar;
}
@end

@implementation TestClass {
@public
  struct { int z; } implementationIvar;
}
@end

//--- Frameworks/Target.framework/Modules/module.modulemap
framework module Target {
  header "Target.h"
  export *
}

//--- Frameworks/Redirect.framework/Headers/Redirect.h
#import <Target/Target.h>

//--- Frameworks/Redirect.framework/Modules/module.modulemap
framework module Redirect {
  header "Redirect.h"
  export *
}

//--- test.m
// At first import everything as non-modular.
#import <Target/Target.h>
// And now as modular to merge same entities obtained through different sources.
#import <Redirect/Redirect.h>
// Non-modular import is achieved through using the same name (-fmodule-name) as the imported framework module.

void test(TestClass *obj) {
  obj->structIvar.left.x = 0;
  obj->unionIvar->y = 1.0f;
  obj->enumIvar = kX;
  int tmp = obj.prop.u;
#ifndef __cplusplus
  tmp += [obj method].v;
#endif

  obj->extensionIvar.y = 0;
  obj->implementationIvar.z = 0;
}
