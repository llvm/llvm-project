## Field Attributes



### counted_by

{clang-attr-syntaxes}`CountedByDocs`

The `counted_by` attribute is applied to a pointer or flexible array member to
indicate that the pointer points to (or the flexible array member contains) at
least the number of *elements* given by the attribute's argument.

This attribute is used by {doc}`-fbounds-safety <BoundsSafety>` to propagate
bounds information on API surfaces without any ABI changes. This attribute is
also used to improve the results of the array bound sanitizer and the
`__builtin_dynamic_object_size` builtin.

Because the size of the pointee type must be known to compute the pointer's
bounds, such a pointer must not be used while its pointee type is incomplete; a
pointer to a forward-declared type is accepted on fields annotated with
`counted_by`, but the type must be completed before the pointer is used. If
the pointee type can never be completed, `counted_by` is rejected and
`sized_by` should be used instead. `void *` is a special case: as a GNU
extension (diagnosed by `-Wgnu-pointer-arith`), `counted_by` is accepted on
it, where it behaves like `sized_by` (the argument is treated as a byte count,
`void` having an assumed size of one byte).

A pointer annotated with `counted_by` must have a count of zero when it is
null. This requirement is currently only enforced when compiling with
{doc}`-fbounds-safety <BoundsSafety>` (see {ref}`Current status of
-fbounds-safety support in upstream Clang <bounds-safety-current-upstream-status>`). Use
`counted_by_or_null` for a pointer that may be null while carrying a nonzero
count.

#### Keeping pointer and count in sync

The `counted_by` attribute establishes a relationship between the annotated
pointer and its count: the pointer must point to at least `count` elements.
Assigning to only one of them can break this relationship.
Without {doc}`-fbounds-safety <BoundsSafety>`, it is the programmer's
responsibility to ensure the pointer and count remain in sync. With
`-fbounds-safety` it is automatically enforced. For example:

```c
struct buffer {
  int *buf __attribute__((counted_by(count)));
  size_t count;
};

void grow(struct buffer *b, size_t new_count) {
  // b->buf isn't updated. The underlying memory pointed to by b->buf might be
  // smaller than new_count which would contradict the counted_by attribute.
  // Compile error with -fbounds-safety but allowed without -fbounds-safety.
  b->count = new_count;
}
```

Updating both together - so that `buf` points to `count` elements - keeps
the attribute true. For example:

```c
void grow(struct buffer *b, size_t new_count) {
  // Allowed by -fbounds-safety
  int *new_buf = malloc(new_count * sizeof(int));
  // -fbounds-safety enforces that the `new_buf` points to at least `new_count`
  // integers at runtime. Without -fbounds-safety nothing enforces this.
  b->buf = new_buf;
  b->count = new_count;
}
```

#### Flexible array members

The `counted_by` attribute may also be applied to the flexible array member of
a structure in C. In this case the argument names the field member holding the
count of elements in the flexible array; that field must be within the same
non-anonymous, enclosing struct as the flexible array member.

This example specifies that the flexible array member `array` has the number
of elements allocated for it in `count`:

```c
struct bar;

struct foo {
  size_t count;
  char other;
  struct bar *array[] __attribute__((counted_by(count)));
};
```

This establishes a relationship between `array` and `count`. Specifically,
`array` must have at least `count` number of elements available. It's the
user's responsibility to ensure that this relationship is maintained through
changes to the structure.

In the following example, the allocated array erroneously has fewer elements
than what's specified by `p->count`. This would result in an out-of-bounds
access not being detected.

```c
#define SIZE_INCR 42

struct foo *p;

void foo_alloc(size_t count) {
  p = malloc(MAX(sizeof(struct foo),
                 offsetof(struct foo, array[0]) + count * sizeof(struct bar *)));
  p->count = count + SIZE_INCR;
}
```

The next example updates `p->count`, but breaks the relationship requirement
that `p->array` must have at least `p->count` number of elements available:

```c
#define SIZE_INCR 42

struct foo *p;

void foo_alloc(size_t count) {
  p = malloc(MAX(sizeof(struct foo),
                 offsetof(struct foo, array[0]) + count * sizeof(struct bar *)));
  p->count = count;
}

void use_foo(int index, int val) {
  p->count += SIZE_INCR + 1; /* 'count' is now larger than the number of elements of 'array'. */
  p->array[index] = val;     /* The sanitizer can't properly check this access. */
}
```

In this example, an update to `p->count` maintains the relationship
requirement:

```c
void use_foo(int index, int val) {
  if (p->count == 0)
    return;
  --p->count;
  p->array[index] = val;
}
```


### counted_by_or_null

{clang-attr-syntaxes}`CountedByOrNullDocs`

The `counted_by_or_null` attribute is applied to a pointer to indicate that,
if the pointer is non-null, it points to memory containing at least the number
of *elements* given by the attribute's argument. If the pointer is null, the
value of the argument is ignored and the pointer points to zero elements.

The `counted_by_or_null` attribute is identical to `counted_by` except that
it treats null pointers differently and cannot be applied to a flexible array
member. Whereas `counted_by` requires a null pointer to have a count of zero,
`counted_by_or_null` allows the pointer to be null regardless of the value of
the count. This supports the common idiom where a pointer is either null or
points to memory containing at least the given number of elements.

Currently only {doc}`-fbounds-safety <BoundsSafety>` makes use of the
distinction between `counted_by_or_null` and `counted_by` (see
{ref}`Current status of -fbounds-safety support in upstream Clang
<bounds-safety-current-upstream-status>`).


### no_unique_address

{clang-attr-syntaxes}`NoUniqueAddressDocs`

The `no_unique_address` attribute allows tail padding in a non-static data
member to overlap other members of the enclosing class (and in the special
case when the type is empty, permits it to fully overlap other members).
The field is laid out as if a base class were encountered at the corresponding
point within the class (except that it does not share a vptr with the enclosing
object).

Example usage:

```c++
template<typename T, typename Alloc> struct my_vector {
  T *p;
  [[no_unique_address]] Alloc alloc;
  // ...
};
static_assert(sizeof(my_vector<int, std::allocator<int>>) == sizeof(int*));
```

`[[no_unique_address]]` is a standard C++20 attribute. Clang supports its use
in C++11 onwards.

On MSVC targets, `[[no_unique_address]]` is ignored; use
`[[msvc::no_unique_address]]` instead. Currently there is no guarantee of ABI
compatibility or stability with MSVC.


(langext-preferred_type_documentation)=

### preferred_type

{clang-attr-syntaxes}`PreferredTypeDocumentation`

This attribute allows adjusting the type of a bit-field in debug information.
This can be helpful when a bit-field is intended to store an enumeration value,
but has to be specified as having the enumeration's underlying type in order to
facilitate compiler optimizations or bit-field packing behavior. Normally, the
underlying type is what is emitted in debug information, which can make it hard
for debuggers to know to map a bit-field's value back to a particular enumeration.

```c++
enum Colors { Red, Green, Blue };

struct S {
  [[clang::preferred_type(Colors)]] unsigned ColorVal : 2;
  [[clang::preferred_type(bool)]] unsigned UseAlternateColorSpace : 1;
} s = { Green, false };
```

Without the attribute, a debugger is likely to display the value `1` for `ColorVal`
and `0` for `UseAlternateColorSpace`. With the attribute, the debugger may now
display `Green` and `false` instead.

This can be used to map a bit-field to an arbitrary type that isn't integral
or an enumeration type. For example:

```c++
struct A {
  short a1;
  short a2;
};

struct B {
  [[clang::preferred_type(A)]] unsigned b1 : 32 = 0x000F'000C;
};
```

will associate the type `A` with the `b1` bit-field and is intended to display
something like this in the debugger:

```text
Process 2755547 stopped
* thread #1, name = 'test-preferred-', stop reason = step in
    frame #0: 0x0000555555555148 test-preferred-type`main at test.cxx:13:14
   10   int main()
   11   {
   12       B b;
-> 13       return b.b1;
   14   }
(lldb) v -T
(B) b = {
  (A:32) b1 = {
    (short) a1 = 12
    (short) a2 = 15
  }
}
```

Note that debuggers may not be able to handle more complex mappings, and so
this usage is debugger-dependent.


### require_explicit_initialization

{clang-attr-syntaxes}`ExplicitInitDocs`

The `clang::require_explicit_initialization` attribute indicates that a
field of an aggregate must be initialized explicitly by the user when an object
of the aggregate type is constructed. The attribute supports both C and C++,
but its usage is invalid on non-aggregates.

Note that this attribute is *not* a memory safety feature, and is *not* intended
to guard against use of uninitialized memory.

Rather, it is intended for use in "parameter-objects", used to simulate,
for example, the passing of named parameters.
Except inside unevaluated contexts, the attribute generates a warning when
explicit initializers for such variables are not provided (this occurs
regardless of whether any in-class field initializers exist):

```c++
struct Buffer {
  void *address [[clang::require_explicit_initialization]];
  size_t length [[clang::require_explicit_initialization]] = 0;
};

struct ArrayIOParams {
  size_t count [[clang::require_explicit_initialization]];
  size_t element_size [[clang::require_explicit_initialization]];
  int flags = 0;
};

size_t ReadArray(FILE *file, struct Buffer buffer,
                 struct ArrayIOParams params);

int main() {
  unsigned int buf[512];
  ReadArray(stdin, {
    buf
    // warning: field 'length' is not explicitly initialized
  }, {
    .count = sizeof(buf) / sizeof(*buf),
    // warning: field 'element_size' is not explicitly initialized
    // (Note that a missing initializer for 'flags' is not diagnosed, because
    // the field is not marked as requiring explicit initialization.)
  });
}
```


### sized_by

{clang-attr-syntaxes}`SizedByDocs`

The `sized_by` attribute is applied to a pointer to indicate that the pointer
points to memory containing at least the number of *bytes* given by the
attribute's argument. It is closely related to `counted_by`; the difference is
that `counted_by` counts the number of *elements* of the pointee type, whereas
`sized_by` counts the number of *bytes*. This makes `sized_by` the natural
choice for `void *` and other byte buffers.

This attribute is used by {doc}`-fbounds-safety <BoundsSafety>` to propagate
bounds information on API surfaces without any ABI changes. This attribute is
also used to improve the results of the array bound sanitizer and the
`__builtin_dynamic_object_size` builtin.

The argument is an expression of integer type, following the same rules as the
argument of `counted_by`. Unlike `counted_by`, `sized_by` cannot be
applied to a C99 flexible array member; it applies to pointers only. For
example:

```c
struct object {
  unsigned long size;
  void *data __attribute__((sized_by(size)));
};
```

A pointer annotated with `sized_by` must have a size of zero when it is null.
This requirement is currently only enforced when compiling with
{doc}`-fbounds-safety <BoundsSafety>` (see {ref}`Current status of
-fbounds-safety support in upstream Clang <bounds-safety-current-upstream-status>`). Use
`sized_by_or_null` for a pointer that may be null while carrying a nonzero
size.

#### Keeping pointer and size in sync

The `sized_by` attribute establishes a relationship between the annotated
pointer and its size: the pointer must point to at least `size` bytes.
Assigning to only one of them can break this relationship.
Without {doc}`-fbounds-safety <BoundsSafety>`, it is the programmer's
responsibility to ensure the pointer and size remain in sync. With
`-fbounds-safety` it is automatically enforced. For example:

```c
struct buffer {
  uint8_t *buf __attribute__((sized_by(size)));
  size_t size;
};

void grow(struct buffer *b, size_t new_size) {
  // b->buf isn't updated. The underlying memory pointed to by b->buf might be
  // smaller than new_size which would contradict the sized_by attribute.
  // Compile error with -fbounds-safety but allowed without -fbounds-safety.
  b->size = new_size;
}
```

Updating both together - so that `buf` points to `size` bytes - keeps
the attribute true. For example:

```c
void grow(struct buffer *b, size_t new_size) {
  // Allowed by -fbounds-safety
  uint8_t *new_buf = malloc(new_size);
  // -fbounds-safety enforces that the `new_buf` points to at least `new_size`
  // bytes at runtime. Without -fbounds-safety nothing enforces this.
  b->buf = new_buf;
  b->size = new_size;
}
```

#### Incomplete and variable-length pointees

`sized_by` is typically applied to `void *` or a pointer to a byte-sized
type, but it may be used with any pointee type. Two situations call for this,
both of which rule out counting fixed-size elements:

First, the pointee type may be incomplete, such as an opaque type. Its element
size is then unavailable, so `counted_by` cannot be used, whereas `sized_by`
bounds the memory in bytes and imposes no completeness requirement.

Second, the buffer may hold variable-length elements, so there is no fixed
element size to count, even though the total byte size is well defined. For
example, a buffer might pack together several structures that each end in a
flexible array member of differing length:

```c
struct var_len {
  int fam_size;
  char data[] __attribute__((counted_by(fam_size)));
};

struct buffer_view {
  int byte_size;
  struct var_len *buf __attribute__((sized_by(byte_size)));
};
```

Here `counted_by` cannot be applied to `buf` because its pointee is a
variable-length structure, but `sized_by` bounds the whole region in bytes;
the region is traversed by advancing a byte offset rather than by indexing
elements.


### sized_by_or_null

{clang-attr-syntaxes}`SizedByOrNullDocs`

The `sized_by_or_null` attribute is applied to a pointer to indicate that, if
the pointer is non-null, it points to memory containing at least the number of
*bytes* given by the attribute's argument. If the pointer is null, the value of
the argument is ignored and the pointer points to zero bytes.

The `sized_by_or_null` attribute is identical to `sized_by` except in how
it treats null pointers. Whereas `sized_by` requires a null pointer to have a
size of zero, `sized_by_or_null` allows the pointer to be null regardless of
the value of the size. This supports the common idiom where a pointer is either
null or points to memory containing at least the given number of bytes.

Currently only {doc}`-fbounds-safety <BoundsSafety>` makes use of the
distinction between `sized_by_or_null` and `sized_by` (see
{ref}`Current status of -fbounds-safety support in upstream Clang
<bounds-safety-current-upstream-status>`).


