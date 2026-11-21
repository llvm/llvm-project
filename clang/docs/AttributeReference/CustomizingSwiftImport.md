## Customizing Swift Import

Clang supports additional attributes for customizing how APIs are imported into
Swift.

### swift_async

{clang-attr-syntaxes}`SwiftAsyncDocs`

The `swift_async` attribute specifies if and how a particular function or
Objective-C method is imported into a swift async method. For instance:

```objc
@interface MyClass : NSObject
-(void)notActuallyAsync:(int)p1 withCompletionHandler:(void (^)())handler
    __attribute__((swift_async(none)));

-(void)actuallyAsync:(int)p1 callThisAsync:(void (^)())fun
    __attribute__((swift_async(swift_private, 1)));
@end
```

Here, `notActuallyAsync:withCompletionHandler` would have been imported as
`async` (because it's last parameter's selector piece is
`withCompletionHandler`) if not for the `swift_async(none)` attribute.
Conversely, `actuallyAsync:callThisAsync` wouldn't have been imported as
`async` if not for the `swift_async` attribute because it doesn't match the
naming convention.

When using `swift_async` to enable importing, the first argument to the
attribute is either `swift_private` or `not_swift_private` to indicate
whether the function/method is private to the current framework, and the second
argument is the index of the completion handler parameter.


### swift_async_error

{clang-attr-syntaxes}`SwiftAsyncErrorDocs`

The `swift_async_error` attribute specifies how an error state will be
represented in a swift async method. It's a bit analogous to the `swift_error`
attribute for the generated async method. The `swift_async_error` attribute
can indicate a variety of different ways of representing an error.

- `__attribute__((swift_async_error(zero_argument, N)))`, specifies that the
  async method is considered to have failed if the Nth argument to the
  completion handler is zero.
- `__attribute__((swift_async_error(nonzero_argument, N)))`, specifies that
  the async method is considered to have failed if the Nth argument to the
  completion handler is non-zero.
- `__attribute__((swift_async_error(nonnull_error)))`, specifies that the
  async method is considered to have failed if the `NSError *` argument to the
  completion handler is non-null.
- `__attribute__((swift_async_error(none)))`, specifies that the async method
  cannot fail.

For instance:

```objc
@interface MyClass : NSObject
-(void)asyncMethod:(void (^)(char, int, float))handler
    __attribute__((swift_async(swift_private, 1)))
    __attribute__((swift_async_error(zero_argument, 2)));
@end
```

Here, the `swift_async` attribute specifies that `handler` is the completion
handler for this method, and the `swift_async_error` attribute specifies that
the `int` parameter is the one that represents the error.


### swift_async_name

{clang-attr-syntaxes}`SwiftAsyncNameDocs`

The `swift_async_name` attribute provides the name of the `async` overload for
the given declaration in Swift. If this attribute is absent, the name is
transformed according to the algorithm built into the Swift compiler.

The argument is a string literal that contains the Swift name of the function or
method. The name may be a compound Swift name. The function or method with such
an attribute must have more than zero parameters, as its last parameter is
assumed to be a callback that's eliminated in the Swift `async` name.

```objc
@interface URL
+ (void) loadContentsFrom:(URL *)url callback:(void (^)(NSData *))data __attribute__((__swift_async_name__("URL.loadContentsFrom(_:)")))
@end
```


### swift_attr

{clang-attr-syntaxes}`SwiftAttrDocs`

The `swift_attr` provides a Swift-specific annotation for the declaration
or type to which the attribute appertains to. It can be used on any declaration
or type in Clang. This kind of annotation is ignored by Clang as it doesn't have any
semantic meaning in languages supported by Clang. The Swift compiler can
interpret these annotations according to its own rules when importing C or
Objective-C declarations.


### swift_bridge

{clang-attr-syntaxes}`SwiftBridgeDocs`

The `swift_bridge` attribute indicates that the declaration to which the
attribute appertains is bridged to the named Swift type.

```objc
__attribute__((__objc_root__))
@interface Base
- (instancetype)init;
@end

__attribute__((__swift_bridge__("BridgedI")))
@interface I : Base
@end
```

In this example, the Objective-C interface `I` will be made available to Swift
with the name `BridgedI`. It would be possible for the compiler to refer to
`I` still in order to bridge the type back to Objective-C.


### swift_bridged

{clang-attr-syntaxes}`SwiftBridgedTypedefDocs`

The `swift_bridged_typedef` attribute indicates that when the typedef to which
the attribute appertains is imported into Swift, it should refer to the bridged
Swift type (e.g. Swift's `String`) rather than the Objective-C type as written
(e.g. `NSString`).

```objc
@interface NSString;
typedef NSString *AliasedString __attribute__((__swift_bridged_typedef__));

extern void acceptsAliasedString(AliasedString _Nonnull parameter);
```

In this case, the function `acceptsAliasedString` will be imported into Swift
as a function which accepts a `String` type parameter.


### swift_error

{clang-attr-syntaxes}`SwiftErrorDocs`

The `swift_error` attribute controls whether a particular function (or
Objective-C method) is imported into Swift as a throwing function, and if so,
which dynamic convention it uses.

All of these conventions except `none` require the function to have an error
parameter. Currently, the error parameter is always the last parameter of type
`NSError**` or `CFErrorRef*`. Swift will remove the error parameter from
the imported API. When calling the API, Swift will always pass a valid address
initialized to a null pointer.

- `swift_error(none)` means that the function should not be imported as
  throwing. The error parameter and result type will be imported normally.
- `swift_error(null_result)` means that calls to the function should be
  considered to have thrown if they return a null value. The return type must be
  a pointer type, and it will be imported into Swift with a non-optional type.
  This is the default error convention for Objective-C methods that return
  pointers.
- `swift_error(zero_result)` means that calls to the function should be
  considered to have thrown if they return a zero result. The return type must be
  an integral type. If the return type would have been imported as `Bool`, it
  is instead imported as `Void`. This is the default error convention for
  Objective-C methods that return a type that would be imported as `Bool`.
- `swift_error(nonzero_result)` means that calls to the function should be
  considered to have thrown if they return a non-zero result. The return type must
  be an integral type. If the return type would have been imported as `Bool`,
  it is instead imported as `Void`.
- `swift_error(nonnull_error)` means that calls to the function should be
  considered to have thrown if they leave a non-null error in the error parameter.
  The return type is left unmodified.


### swift_name

{clang-attr-syntaxes}`SwiftNameDocs`

The `swift_name` attribute provides the name of the declaration in Swift. If
this attribute is absent, the name is transformed according to the algorithm
built into the Swift compiler.

The argument is a string literal that contains the Swift name of the function,
variable, or type. When renaming a function, the name may be a compound Swift
name. For a type, enum constant, property, or variable declaration, the name
must be a simple or qualified identifier.

```objc
@interface URL
- (void) initWithString:(NSString *)s __attribute__((__swift_name__("URL.init(_:)")))
@end

void __attribute__((__swift_name__("squareRoot()"))) sqrt(double v) {
}
```


### swift_newtype

{clang-attr-syntaxes}`SwiftNewTypeDocs`

The `swift_newtype` attribute indicates that the typedef to which the
attribute appertains is imported as a new Swift type of the typedef's name.
Previously, the attribute was spelt `swift_wrapper`. While the behaviour of
the attribute is identical with either spelling, `swift_wrapper` is
deprecated, only exists for compatibility purposes, and should not be used in
new code.

- `swift_newtype(struct)` means that a Swift struct will be created for this
  typedef.

- `swift_newtype(enum)` means that a Swift enum will be created for this
  typedef.

  ```c
  // Import UIFontTextStyle as an enum type, with enumerated values being
  // constants.
  typedef NSString * UIFontTextStyle __attribute__((__swift_newtype__(enum)));

  // Import UIFontDescriptorFeatureKey as a structure type, with enumerated
  // values being members of the type structure.
  typedef NSString * UIFontDescriptorFeatureKey __attribute__((__swift_newtype__(struct)));
  ```


### swift_objc_members

{clang-attr-syntaxes}`SwiftObjCMembersDocs`

This attribute indicates that Swift subclasses and members of Swift extensions
of this class will be implicitly marked with the `@objcMembers` Swift
attribute, exposing them back to Objective-C.


### swift_private

{clang-attr-syntaxes}`SwiftPrivateDocs`

Declarations marked with the `swift_private` attribute are hidden from the
framework client but are still made available for use within the framework or
Swift SDK overlay.

The purpose of this attribute is to permit a more idomatic implementation of
declarations in Swift while hiding the non-idiomatic one.


