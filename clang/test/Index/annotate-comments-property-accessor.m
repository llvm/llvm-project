// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://12378879

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

@interface AppDelegate
/**
  \brief This is ReadonlyProperty
*/
@property (readonly, getter = ReadonlyGetter) int MyProperty;
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-1]]" column="51"><Name>MyProperty</Name><USR>c:objc(cs)AppDelegate(py)MyProperty</USR><Declaration>- (int)ReadonlyGetter;</Declaration><Abstract><Para> This is ReadonlyProperty</Para></Abstract></Function>]

/**
  \brief This is GeneralProperty
*/
@property int GeneralProperty;
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-1]]" column="15"><Name>GeneralProperty</Name><USR>c:objc(cs)AppDelegate(py)GeneralProperty</USR><Declaration>- (int)GeneralProperty;</Declaration><Abstract><Para> This is GeneralProperty</Para></Abstract></Function>]
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-2]]" column="15"><Name>GeneralProperty</Name><USR>c:objc(cs)AppDelegate(py)GeneralProperty</USR><Declaration>- (void)setGeneralProperty:(int)GeneralProperty;</Declaration><Abstract><Para> This is GeneralProperty</Para></Abstract></Function>]

/**
  \brief This is PropertyInPrimaryClass
*/
@property (copy, nonatomic) id PropertyInPrimaryClass;
- (void) setThisRecord : (id)arg;
- (id) Record;
@end
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-4]]" column="32"><Name>PropertyInPrimaryClass</Name><USR>c:objc(cs)AppDelegate(py)PropertyInPrimaryClass</USR><Declaration>- (id)PropertyInPrimaryClass;</Declaration><Abstract><Para> This is PropertyInPrimaryClass</Para></Abstract></Function>]
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-5]]" column="32"><Name>PropertyInPrimaryClass</Name><USR>c:objc(cs)AppDelegate(py)PropertyInPrimaryClass</USR><Declaration>- (void)setPropertyInPrimaryClass:(id)PropertyInPrimaryClass;</Declaration><Abstract><Para> This is PropertyInPrimaryClass</Para></Abstract></Function>]

@interface AppDelegate()
- (id) GetterInClassExtension;
/**
  \brief This is Record
*/
@property (copy, setter = setThisRecord:) id Record;
@end
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-6]]" column="1"><Name>PropertyInClassExtension</Name><USR>c:objc(cs)AppDelegate(py)PropertyInClassExtension</USR><Declaration>- (id)GetterInClassExtension;</Declaration><Abstract><Para> This is PropertyInClassExtension</Para></Abstract></Function>]

@interface AppDelegate()
/**
  \brief This is PropertyInClassExtension
*/
@property (copy, getter = GetterInClassExtension) id PropertyInClassExtension;

- (id) PropertyInPrimaryClass;
@end
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-4]]" column="54"><Name>PropertyInClassExtension</Name><USR>c:objc(cs)AppDelegate(py)PropertyInClassExtension</USR><Declaration>- (id)GetterInClassExtension;</Declaration><Abstract><Para> This is PropertyInClassExtension</Para></Abstract></Function>]
// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments-property-accessor.m" line="[[@LINE-5]]" column="54"><Name>PropertyInClassExtension</Name><USR>c:objc(cs)AppDelegate(py)PropertyInClassExtension</USR><Declaration>- (void)setPropertyInClassExtension:(id)PropertyInClassExtension;</Declaration><Abstract><Para> This is PropertyInClassExtension</Para></Abstract></Function>]
  
@implementation AppDelegate
- (id) PropertyInPrimaryClass { return 0; }
@end





