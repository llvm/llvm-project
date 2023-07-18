// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

// rdar://12378714

/**
 * \brief This is a protocol definition
*/
@protocol MyProto
@optional
/**
 * \brief MethodMyProto method
 * \param[in] anObject input value
 * \param[in] range output value is unsigned int
 * \result return index
 */
- (unsigned int)MethodMyProto:(nullable id)anObject inRange:(unsigned int)range;
/**
 * \brief PropertyMyProto - This is protocol's property.
*/
@property (copy, nonnull) id PropertyMyProto;
/**
 * \brief ClassMethodMyProto
*/
+ ClassMethodMyProto;
@end
// CHECK: <Declaration>@protocol MyProto\n@end</Declaration>
// CHECK: <Declaration>- (unsigned int)MethodMyProto:(nullable id)anObject inRange:(unsigned int)range;</Declaration>
// CHECK: <Declaration>@optional\n@property(atomic, copy, readwrite, nonnull) id PropertyMyProto;</Declaration>
// CHECK: <Declaration>+ (id)ClassMethodMyProto;</Declaration>

/**
 * \brief NSObject is the root class.
*/
@interface NSObject {
/**
 * \brief IvarNSObject
*/
  id IvarNSObject;
}
@end
// CHECK: Declaration>@interface NSObject {\n  id IvarNSObject;\n}\n@end</Declaration>
// CHECK: <Declaration>id IvarNSObject</Declaration>

/**
 * \brief MyClass - primary class.
*/
@interface MyClass : NSObject<MyProto>
{
/**
 * \brief IvarMyClass - IvarMyClass of values.
*/
  id IvarMyClass;
}
/**
 * \brief MethodMyClass is instance method.
*/
- MethodMyClass;

/**
 * \brief ClassMethodMyClass is class method.
*/
+ ClassMethodMyClass;

/**
 * \brief PropertyMyClass - This is class's property.
*/
@property (copy) id PropertyMyClass;
@end
// CHECK: <Declaration>@interface MyClass : NSObject &lt;MyProto&gt; {\n    id IvarMyClass;\n}\n@end</Declaration>
// CHECK: <Declaration>id IvarMyClass</Declaration>
// CHECK: <Declaration>- (id)MethodMyClass;</Declaration>
// CHECK: <Declaration>+ (id)ClassMethodMyClass;</Declaration>
// CHECK: <Declaration>@property(atomic, copy, readwrite) id PropertyMyClass;</Declaration

/**
 * \brief - This is class extension of MyClass
*/
@interface MyClass()
{
/**
 * \brief IvarMyClassExtension - IvarMyClassExtension private to class extension
*/
  id IvarMyClassExtension;
}
@end
// CHECK: <Declaration>@interface MyClass () {\n  id IvarMyClassExtension;\n}\n@end</Declaration>
// CHECK: <Declaration>id IvarMyClassExtension</Declaration>


/**
 * \brief MyClass (Category) is private to MyClass.
*/
@interface MyClass (Category)
/**
 * \brief This is private to MyClass
 */
- (void)MethodMyClassCategory;

/**
 * \brief PropertyMyClassCategory - This is class's private property.
*/
@property (copy) id PropertyMyClassCategory;
@end
// CHECK: <Declaration>@interface MyClass (Category)\n@end</Declaration>
// CHECK: <Declaration>- (void)MethodMyClassCategory;</Declaration>
// CHECK: <Declaration>@property(atomic, copy, readwrite) id PropertyMyClassCategory;</Declaration>
// CHECK: <Declaration>- (id)PropertyMyClassCategory;</Declaration>
// CHECK: <Declaration>- (void)setPropertyMyClassCategory:(id)arg;</Declaration>

/// @implementation's

/**
 * \brief implementation of MyClass class.
*/
@implementation MyClass {
/**
 * \brief IvarPrivateToMyClassImpl.
*/
  id IvarPrivateToMyClassImpl;
}
/**
 * \brief MethodMyClass is instance method implementation.
*/
- MethodMyClass {
  return 0;
}

/**
 * \brief ClassMethodMyClass is class method implementation.
*/
+ ClassMethodMyClass {
  return 0;
}
@end
// CHECK: <Declaration>@implementation MyClass {\n  id IvarPrivateToMyClassImpl;\n  id _PropertyMyClass;\n}\n@end</Declaration>
// CHECK: <Declaration>id IvarPrivateToMyClassImpl</Declaration>
// CHECK: <Declaration>- (id)MethodMyClass;</Declaration>
// CHECK: <Declaration>+ (id)ClassMethodMyClass;</Declaration>

/**
 * \brief MyClass (Category) is implementation of private to MyClass.
*/
@implementation MyClass (Category)
/**
 * \brief This is private to MyClass
 */
- (void)MethodMyClassCategory {}
/**
 * \brief property getter
*/
- (id) PropertyMyClassCategory { return 0; }

/**
 * \brief property setter
*/
- (void) setPropertyMyClassCategory : (id) arg {}
@end
// CHECK: <Declaration>@implementation MyClass (Category)\n@end</Declaration>
// CHECK: <Declaration>- (void)MethodMyClassCategory;</Declaration>
// CHECK: <Declaration>- (id)PropertyMyClassCategory;</Declaration>
// CHECK: <Declaration>- (void)setPropertyMyClassCategory:(id)arg;</Declaration>

/**
 * \brief NSObject implementation
*/
@implementation NSObject
@end
// CHECK: <Declaration>@implementation NSObject\n@end</Declaration>
