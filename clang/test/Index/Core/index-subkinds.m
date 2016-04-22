// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

// CHECK: [[@LINE+1]]:12 | class/ObjC | XCTestCase | c:objc(cs)XCTestCase | _OBJC_CLASS_$_XCTestCase | Decl | rel: 0
@interface XCTestCase
@end

// CHECK: [[@LINE+1]]:12 | class(test)/ObjC | MyTestCase | c:objc(cs)MyTestCase | _OBJC_CLASS_$_MyTestCase | Decl | rel: 0
@interface MyTestCase : XCTestCase
@end
// CHECK: [[@LINE+1]]:17 | class(test)/ObjC | MyTestCase | c:objc(cs)MyTestCase | <no-cgname> | Def | rel: 0
@implementation MyTestCase
// CHECK: [[@LINE+1]]:1 | instance-method(test)/ObjC | testMe | c:objc(cs)MyTestCase(im)testMe | -[MyTestCase testMe] | Def,Dyn,RelChild | rel: 1
-(void)testMe {}
// CHECK: [[@LINE+1]]:1 | instance-method/ObjC | testResult | c:objc(cs)MyTestCase(im)testResult | -[MyTestCase testResult] | Def,Dyn,RelChild | rel: 1
-(id)testResult { return 0; }
// CHECK: [[@LINE+1]]:1 | instance-method/ObjC | testWithInt: | c:objc(cs)MyTestCase(im)testWithInt: | -[MyTestCase testWithInt:] | Def,Dyn,RelChild | rel: 1
-(void)testWithInt:(int)i {}
@end

// CHECK: [[@LINE+1]]:12 | class(test)/ObjC | SubTestCase | c:objc(cs)SubTestCase | _OBJC_CLASS_$_SubTestCase | Decl | rel: 0
@interface SubTestCase : MyTestCase
@end
// CHECK: [[@LINE+1]]:17 | class(test)/ObjC | SubTestCase | c:objc(cs)SubTestCase | <no-cgname> | Def | rel: 0
@implementation SubTestCase
// CHECK: [[@LINE+1]]:1 | instance-method(test)/ObjC | testIt2 | c:objc(cs)SubTestCase(im)testIt2 | -[SubTestCase testIt2] | Def,Dyn,RelChild | rel: 1
-(void)testIt2 {}
@end

// CHECK: [[@LINE+1]]:12 | extension/ObjC | cat | c:objc(cy)MyTestCase@cat | <no-cgname> | Decl | rel: 0
@interface MyTestCase(cat)
@end
// CHECK: [[@LINE+1]]:17 | extension/ObjC | MyTestCase | c:objc(cy)MyTestCase@cat | <no-cgname> | Def | rel: 0
@implementation MyTestCase(cat)
// CHECK: [[@LINE+1]]:1 | instance-method(test)/ObjC | testInCat | c:objc(cs)MyTestCase(im)testInCat | -[MyTestCase(cat) testInCat] | Def,Dyn,RelChild | rel: 1
- (void)testInCat {}
@end


@class NSButton;
@interface IBCls
// CHECK: [[@LINE+2]]:34 | instance-method/ObjC | prop | c:objc(cs)IBCls(im)prop | -[IBCls prop] | Decl,Dyn,RelChild | rel: 1
// CHECK: [[@LINE+1]]:34 | instance-property(IB)/ObjC | prop | c:objc(cs)IBCls(py)prop | <no-cgname> | Decl,RelChild | rel: 1
@property (readonly) IBOutlet id prop;
// CHECK: [[@LINE+1]]:54 | instance-property(IB,IBColl)/ObjC | propColl | c:objc(cs)IBCls(py)propColl | <no-cgname> | Decl,RelChild | rel: 1
@property (readonly) IBOutletCollection(NSButton) id propColl;
// CHECK: [[@LINE+1]]:1 | instance-method(IB)/ObjC | doIt | c:objc(cs)IBCls(im)doIt | -[IBCls doIt] | Decl,Dyn,RelChild | rel: 1
-(IBAction)doIt;
@end
