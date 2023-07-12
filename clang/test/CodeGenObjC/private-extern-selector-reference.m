// RUN: %clang_cc1 -triple x86_64-apple-ios6.0.0 -emit-llvm -o - %s | FileCheck %s

@interface Query
+ (void)_configureCI;
@end

__attribute__((visibility("default"))) __attribute__((availability(ios,introduced=7.0)))
@interface ObserverQuery : Query @end

@implementation ObserverQuery
+ (void)_configureCI {
    [super _configureCI];
}
@end

// CHECK: @"OBJC_METACLASS_$_ObserverQuery" = global %struct._class_t
// CHECK: @OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global
