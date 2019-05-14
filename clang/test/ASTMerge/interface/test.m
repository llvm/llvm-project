// FIXME: Errors are now warnings.
// XFAIL: *
// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/interface1.m
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/interface2.m
// RUN: not %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: interface2.m:16:9: error: instance variable 'ivar2' declared with incompatible types in different translation units ('float' vs. 'int')
// CHECK: interface1.m:16:7: note: declared here with type 'int'
// CHECK: interface1.m:21:12: error: class 'I4' has incompatible superclasses
// CHECK: interface1.m:21:17: note: inherits from superclass 'I2' here
// CHECK: interface2.m:21:17: note: inherits from superclass 'I1' here
// CHECK: interface2.m:33:1: error: class method 'foo' has incompatible result types in different translation units ('float' vs. 'int')
// CHECK: interface1.m:34:1: note: class method 'foo' also declared here
// CHECK: interface2.m:39:19: error: class method 'bar:' has a parameter with a different types in different translation units ('float' vs. 'int')
// CHECK: interface1.m:40:17: note: declared here with type 'int'
// CHECK: interface2.m:45:1: error: class method 'bar:' is variadic in one translation unit and not variadic in another
// CHECK: interface1.m:46:1: note: class method 'bar:' also declared here
// CHECK: interface2.m:57:20: error: instance method 'bar:' has a parameter with a different types in different translation units ('double' vs. 'float')
// CHECK: interface1.m:58:19: note: declared here with type 'float'
// CHECK: interface1.m:100:17: error: class 'I15' has incompatible superclasses
// CHECK: interface1.m:100:17: note: inherits from superclass 'I12' here
// CHECK: interface2.m:99:17: note: inherits from superclass 'I11' here
// CHECK: 8 errors generated

