// RUN: %clang_cc1 %s 2>&1 | FileCheck %s

// Just shouldn't crash. -verify suppresses the crash, so don't use it.
// PR13417
// CHECK-NOT: Assertion failed
@interface ExtensionActionContextMenu @end
@implementation ExtensionActionContextMenu
namespace {
