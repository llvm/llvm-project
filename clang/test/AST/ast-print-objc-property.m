// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

@interface NSObject
@end

@interface Properties : NSObject
@property(class) int classFoo;
@property(nonatomic) int atomicBar;
@property(readonly) int readonlyConstant;
@property(retain, nonatomic, setter=my_setter:, getter=my_getter) id                   __crazy_name;
@property(nonatomic, strong, nullable) NSObject *                   objProperty;
@property(nonatomic, weak, null_resettable) NSObject *   weakObj;
@property(nonatomic, copy, nonnull) NSObject * copyObj;
@end

// CHECK: @property(class, atomic, assign, unsafe_unretained, readwrite) int classFoo;
// CHECK: @property(nonatomic, assign, unsafe_unretained, readwrite) int atomicBar;
// CHECK: @property(atomic, readonly) int readonlyConstant;
// CHECK: @property(nonatomic, retain, readwrite, getter = my_getter, setter = my_setter:) id __crazy_name;
// CHECK: @property(nonatomic, strong, readwrite, nullable) NSObject *objProperty;
// CHECK: @property(nonatomic, weak, readwrite, null_resettable) NSObject *weakObj;
// CHECK: @property(nonatomic, copy, readwrite, nonnull) NSObject *copyObj;
