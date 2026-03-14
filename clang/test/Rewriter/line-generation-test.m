// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -fms-extensions -rewrite-objc -debug-info-kind=limited %t.mm -o %t-rw.cpp
// RUN: FileCheck  -check-prefix CHECK-LINE --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fms-extensions -rewrite-objc %t.mm -o %t-rwnog.cpp
// RUN: FileCheck  -check-prefix CHECK-NOLINE --input-file=%t-rwnog.cpp %s

__attribute__((objc_root_class)) @interface MyObject {
@public
    id _myLeader;
    id _isTickledPink;
}
@property(retain) id myLeader;
@property(assign) id isTickledPink;
@end

@implementation MyObject

@synthesize myLeader = _myLeader;
@synthesize isTickledPink = _isTickledPink;

- (void) doSomething {
    _myLeader = _isTickledPink;
}

@end

MyObject * foo ()
{
	MyObject* p;
        p.isTickledPink = p.myLeader;	// ok
	p->_isTickledPink = p->_myLeader;
	return p->_isTickledPink;
}

// CHECK-LINE: #line 21
// CHECK-LINE: #line 27
// CHECK-NOLINE-NOT: #line 21
// CHECK-NOLINE-NOT: #line 27

