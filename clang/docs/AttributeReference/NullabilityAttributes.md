## Nullability Attributes

Whether a particular pointer may be "null" is an important concern when working
with pointers in the C family of languages. The various nullability attributes
indicate whether a particular pointer can be null or not, which makes APIs more
expressive and can help static analysis tools identify bugs involving null
pointers. Clang supports several kinds of nullability attributes: the
`nonnull` and `returns_nonnull` attributes indicate which function or
method parameters and result types can never be null, while nullability type
qualifiers indicate which pointer types can be null (`_Nullable`) or cannot
be null (`_Nonnull`).

The nullability (type) qualifiers express whether a value of a given pointer
type can be null (the `_Nullable` qualifier), doesn't have a defined meaning
for null (the `_Nonnull` qualifier), or for which the purpose of null is
unclear (the `_Null_unspecified` qualifier). Because nullability qualifiers
are expressed within the type system, they are more general than the
`nonnull` and `returns_nonnull` attributes, allowing one to express (for
example) a nullable pointer to an array of nonnull pointers. Nullability
qualifiers are written to the right of the pointer to which they apply. For
example:

```c
// No meaningful result when 'ptr' is null (here, it happens to be undefined behavior).
int fetch(int * _Nonnull ptr) { return *ptr; }

// 'ptr' may be null.
int fetch_or_zero(int * _Nullable ptr) {
  return ptr ? *ptr : 0;
}

// A nullable pointer to non-null pointers to const characters.
const char *join_strings(const char * _Nonnull * _Nullable strings, unsigned n);
```

In Objective-C, there is an alternate spelling for the nullability qualifiers
that can be used in Objective-C methods and properties using context-sensitive,
non-underscored keywords. For example:

```objective-c
@interface NSView : NSResponder
  - (nullable NSView *)ancestorSharedWithView:(nonnull NSView *)aView;
  @property (assign, nullable) NSView *superview;
  @property (readonly, nonnull) NSArray *subviews;
@end
```

As well as built-in pointer types, the nullability attributes can be attached
to C++ classes marked with the `_Nullable` attribute.

The following C++ standard library types are considered nullable:
`unique_ptr`, `shared_ptr`, `auto_ptr`, `exception_ptr`, `function`,
`move_only_function` and `coroutine_handle`.

Types should be marked nullable only where the type itself leaves nullability
ambiguous. For example, `std::optional` is not marked `_Nullable`, because
`optional<int> _Nullable` is redundant and `optional<int> _Nonnull` is
not a useful type. `std::weak_ptr` is not nullable, because its nullability
can change with no visible modification, so static annotation is unlikely to be
unhelpful.

### _Nonnull

{clang-attr-syntaxes}`TypeNonNullDocs`

The `_Nonnull` nullability qualifier indicates that null is not a meaningful
value for a value of the `_Nonnull` pointer type. For example, given a
declaration such as:

```c
int fetch(int * _Nonnull ptr);
```

a caller of `fetch` should not provide a null value, and the compiler will
produce a warning if it sees a literal null value passed to `fetch`. Note
that, unlike the declaration attribute `nonnull`, the presence of
`_Nonnull` does not imply that passing null is undefined behavior: `fetch`
is free to consider null undefined behavior or (perhaps for
backward-compatibility reasons) defensively handle null.


### _Null_unspecified

{clang-attr-syntaxes}`TypeNullUnspecifiedDocs`

The `_Null_unspecified` nullability qualifier indicates that neither the
`_Nonnull` nor `_Nullable` qualifiers make sense for a particular pointer
type. It is used primarily to indicate that the role of null with specific
pointers in a nullability-annotated header is unclear, e.g., due to
overly-complex implementations or historical factors with a long-lived API.


### _Nullable

{clang-attr-syntaxes}`TypeNullableDocs`

The `_Nullable` nullability qualifier indicates that a value of the
`_Nullable` pointer type can be null. For example, given:

```c
int fetch_or_zero(int * _Nullable ptr);
```

a caller of `fetch_or_zero` can provide null.

The `_Nullable` attribute on classes indicates that the given class can
represent null values, and so the `_Nullable`, `_Nonnull` etc qualifiers
make sense for this type. For example:

```c
class _Nullable ArenaPointer { ... };

ArenaPointer _Nonnull x = ...;
ArenaPointer _Nullable y = nullptr;
```


### _Nullable_result

{clang-attr-syntaxes}`TypeNullableResultDocs`

The `_Nullable_result` nullability qualifier means that a value of the
`_Nullable_result` pointer can be `nil`, just like `_Nullable`. Where this
attribute differs from `_Nullable` is when it's used on a parameter to a
completion handler in a Swift async method. For instance, here:

```objc
-(void)fetchSomeDataWithID:(int)identifier
         completionHandler:(void (^)(Data *_Nullable_result result, NSError *error))completionHandler;
```

This method asynchronously calls `completionHandler` when the data is
available, or calls it with an error. `_Nullable_result` indicates to the
Swift importer that this is the uncommon case where `result` can get `nil`
even if no error has occurred, and will therefore import it as a Swift optional
type. Otherwise, if `result` was annotated with `_Nullable`, the Swift
importer will assume that `result` will always be non-nil unless an error
occurred.


### nonnull

{clang-attr-syntaxes}`NonNullDocs`

The `nonnull` attribute indicates that some function parameters must not be
null, and can be used in several different ways. It's original usage
([from GCC](https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes))
is as a function (or Objective-C method) attribute that specifies which
parameters of the function are nonnull in a comma-separated list. For example:

```c
extern void * my_memcpy (void *dest, const void *src, size_t len)
                __attribute__((nonnull (1, 2)));
```

Here, the `nonnull` attribute indicates that parameters 1 and 2
cannot have a null value. Omitting the parenthesized list of parameter indices
means that all parameters of pointer type cannot be null:

```c
extern void * my_memcpy (void *dest, const void *src, size_t len)
                __attribute__((nonnull));
```

Clang also allows the `nonnull` attribute to be placed directly on a function
(or Objective-C method) parameter, eliminating the need to specify the
parameter index ahead of type. For example:

```c
extern void * my_memcpy (void *dest __attribute__((nonnull)),
                         const void *src __attribute__((nonnull)), size_t len);
```

Note that the `nonnull` attribute indicates that passing null to a non-null
parameter is undefined behavior, which the optimizer may take advantage of to,
e.g., remove null checks. The `_Nonnull` type qualifier indicates that a
pointer cannot be null in a more general manner (because it is part of the type
system) and does not imply undefined behavior, making it more widely applicable.


### returns_nonnull

{clang-attr-syntaxes}`ReturnsNonNullDocs`

The `returns_nonnull` attribute indicates that a particular function (or
Objective-C method) always returns a non-null pointer. For example, a
particular system `malloc` might be defined to terminate a process when
memory is not available rather than returning a null pointer:

```c
extern void * malloc (size_t size) __attribute__((returns_nonnull));
```

The `returns_nonnull` attribute implies that returning a null pointer is
undefined behavior, which the optimizer may take advantage of. The `_Nonnull`
type qualifier indicates that a pointer cannot be null in a more general manner
(because it is part of the type system) and does not imply undefined behavior,
making it more widely applicable


