// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test-functions.m \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test a case when Objective-C interface ivars are present in two different modules.

//--- Frameworks/Foundation.framework/Headers/Foundation.h
@interface NSObject
@end

//--- Frameworks/Foundation.framework/Modules/module.modulemap
framework module Foundation {
  header "Foundation.h"
  export *
}

//--- Frameworks/ObjCInterface.framework/Headers/ObjCInterface.h
#import <Foundation/Foundation.h>
@interface ObjCInterface : NSObject {
@public
  id _item;
}
@end

@interface WithBitFields : NSObject {
@public
  int x: 3;
  int y: 4;
}
@end

//--- Frameworks/ObjCInterface.framework/Modules/module.modulemap
framework module ObjCInterface {
  header "ObjCInterface.h"
  export *
}

//--- Frameworks/ObjCInterfaceCopy.framework/Headers/ObjCInterfaceCopy.h
#import <Foundation/Foundation.h>
@interface ObjCInterface : NSObject {
@public
  id _item;
}
@end

@interface WithBitFields : NSObject {
@public
  int x: 3;
  int y: 4;
}
@end

// Inlined function present only in Copy.framework to make sure it uses decls from Copy module.
__attribute__((always_inline)) void inlinedIVarAccessor(ObjCInterface *obj, WithBitFields *bitFields) {
  obj->_item = 0;
  bitFields->x = 0;
}

//--- Frameworks/ObjCInterfaceCopy.framework/Modules/module.modulemap
framework module ObjCInterfaceCopy {
  header "ObjCInterfaceCopy.h"
  export *
}

//--- test.m
#import <ObjCInterface/ObjCInterface.h>
#import <ObjCInterfaceCopy/ObjCInterfaceCopy.h>

@implementation ObjCInterface
- (void)test:(id)item {
  _item = item;
}
@end

@implementation WithBitFields
- (void)reset {
  x = 0;
  y = 0;
}
@end

//--- test-functions.m
#import <ObjCInterface/ObjCInterface.h>

void testAccessIVar(ObjCInterface *obj, id item) {
  obj->_item = item;
}
void testAccessBitField(WithBitFields *obj) {
  obj->x = 0;
}

#import <ObjCInterfaceCopy/ObjCInterfaceCopy.h>

void testAccessIVarLater(ObjCInterface *obj, id item) {
  obj->_item = item;
}
void testAccessBitFieldLater(WithBitFields *obj) {
  obj->y = 0;
}
void testInlinedFunction(ObjCInterface *obj, WithBitFields *bitFields) {
  inlinedIVarAccessor(obj, bitFields);
}
