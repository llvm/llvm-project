// RUN: %clang_cc1 -S -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Checks debug info for properties from class extensions for a few cases.


// Readonly property in interface made readwrite in a category, with @impl
// The interesting bit is that when the ivar debug info is generated, the corresponding
// property is looked up and also gets debug info. If the debug info from the interface's
// declaration and from the ivar doesn't match, this will end up with two DIObjCProperty
// entries which would be bad.
@interface FooROWithImpl
// CHECK-NOT: !DIObjCProperty(name: "evolvingpropwithimpl"{{.*}}line: [[@LINE+1]]
@property (readonly) int evolvingpropwithimpl;
@end
@interface FooROWithImpl ()
// CHECK: !DIObjCProperty(name: "evolvingpropwithimpl"{{.*}}line: [[@LINE+1]]
@property int evolvingpropwithimpl;
@end
@implementation FooROWithImpl
@synthesize evolvingpropwithimpl = _evolvingpropwithimpl;
@end


// Simple property from a class extension:
@interface Foo
@end
@interface Foo()
// CHECK: !DIObjCProperty(name: "myprop"{{.*}}line: [[@LINE+1]]
@property int myprop;
@end
// There's intentionally no @implementation for Foo, because that would
// generate debug info for the property via the backing ivar.


// Readonly property in interface made readwrite in a category:
@interface FooRO
// Shouldn't be here but in the class extension below.
// CHECK-NOT: !DIObjCProperty(name: "evolvingprop"{{.*}}line: [[@LINE+1]]
@property (readonly) int evolvingprop;
@end
@interface FooRO ()
// CHECK: !DIObjCProperty(name: "evolvingprop"{{.*}}line: [[@LINE+1]]
@property int evolvingprop;
@end


// This references types in this file to force emission of their debug info.
void foo(Foo *f, FooRO *g, FooROWithImpl* h) { }
