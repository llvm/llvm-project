## Type Safety Checking

Clang supports additional attributes to enable checking type safety properties
that can't be enforced by the C type system. To see warnings produced by these
checks, ensure that -Wtype-safety is enabled. Use cases include:

- MPI library implementations, where these attributes enable checking that
  the buffer type matches the passed `MPI_Datatype`;
- for HDF5 library there is a similar use case to MPI;
- checking types of variadic functions' arguments for functions like
  `fcntl()` and `ioctl()`.

You can detect support for these attributes with `__has_attribute()`. For
example:

```c++
#if defined(__has_attribute)
#  if __has_attribute(argument_with_type_tag) && \
      __has_attribute(pointer_with_type_tag) && \
      __has_attribute(type_tag_for_datatype)
#    define ATTR_MPI_PWT(buffer_idx, type_idx) __attribute__((pointer_with_type_tag(mpi,buffer_idx,type_idx)))
/* ... other macros ... */
#  endif
#endif

#if !defined(ATTR_MPI_PWT)
# define ATTR_MPI_PWT(buffer_idx, type_idx)
#endif

int MPI_Send(void *buf, int count, MPI_Datatype datatype /*, other args omitted */)
    ATTR_MPI_PWT(1,3);
```

### argument_with_type_tag

{clang-attr-syntaxes}`ArgumentWithTypeTagDocs`

Use `__attribute__((argument_with_type_tag(arg_kind, arg_idx,
type_tag_idx)))` on a function declaration to specify that the function
accepts a type tag that determines the type of some other argument.

This attribute is primarily useful for checking arguments of variadic functions
(`pointer_with_type_tag` can be used in most non-variadic cases).

In the attribute prototype above:
: - `arg_kind` is an identifier that should be used when annotating all
    applicable type tags.
  - `arg_idx` provides the position of a function argument. The expected type of
    this function argument will be determined by the function argument specified
    by `type_tag_idx`. In the code example below, "3" means that the type of the
    function's third argument will be determined by `type_tag_idx`.
  - `type_tag_idx` provides the position of a function argument. This function
    argument will be a type tag. The type tag will determine the expected type of
    the argument specified by `arg_idx`. In the code example below, "2" means
    that the type tag associated with the function's second argument should agree
    with the type of the argument specified by `arg_idx`.

For example:

```c++
int fcntl(int fd, int cmd, ...)
    __attribute__(( argument_with_type_tag(fcntl,3,2) ));
// The function's second argument will be a type tag; this type tag will
// determine the expected type of the function's third argument.
```


### pointer_with_type_tag

{clang-attr-syntaxes}`PointerWithTypeTagDocs`

Use `__attribute__((pointer_with_type_tag(ptr_kind, ptr_idx, type_tag_idx)))`
on a function declaration to specify that the function accepts a type tag that
determines the pointee type of some other pointer argument.

In the attribute prototype above:
: - `ptr_kind` is an identifier that should be used when annotating all
    applicable type tags.
  - `ptr_idx` provides the position of a function argument; this function
    argument will have a pointer type. The expected pointee type of this pointer
    type will be determined by the function argument specified by
    `type_tag_idx`. In the code example below, "1" means that the pointee type
    of the function's first argument will be determined by `type_tag_idx`.
  - `type_tag_idx` provides the position of a function argument; this function
    argument will be a type tag. The type tag will determine the expected pointee
    type of the pointer argument specified by `ptr_idx`. In the code example
    below, "3" means that the type tag associated with the function's third
    argument should agree with the pointee type of the pointer argument specified
    by `ptr_idx`.

For example:

```c++
typedef int MPI_Datatype;
int MPI_Send(void *buf, int count, MPI_Datatype datatype /*, other args omitted */)
    __attribute__(( pointer_with_type_tag(mpi,1,3) ));
// The function's 3rd argument will be a type tag; this type tag will
// determine the expected pointee type of the function's 1st argument.
```


### type_tag_for_datatype

{clang-attr-syntaxes}`TypeTagForDatatypeDocs`

When declaring a variable, use
`__attribute__((type_tag_for_datatype(kind, type)))` to create a type tag that
is tied to the `type` argument given to the attribute.

In the attribute prototype above:
: - `kind` is an identifier that should be used when annotating all applicable
    type tags.
  - `type` indicates the name of the type.

Clang supports annotating type tags of two forms.

- **Type tag that is a reference to a declared identifier.**
  Use `__attribute__((type_tag_for_datatype(kind, type)))` when declaring that
  identifier:

  ```c++
  typedef int MPI_Datatype;
  extern struct mpi_datatype mpi_datatype_int
      __attribute__(( type_tag_for_datatype(mpi,int) ));
  #define MPI_INT ((MPI_Datatype) &mpi_datatype_int)
  // &mpi_datatype_int is a type tag. It is tied to type "int".
  ```

- **Type tag that is an integral literal.**
  Declare a `static const` variable with an initializer value and attach
  `__attribute__((type_tag_for_datatype(kind, type)))` on that declaration:

  ```c++
  typedef int MPI_Datatype;
  static const MPI_Datatype mpi_datatype_int
      __attribute__(( type_tag_for_datatype(mpi,int) )) = 42;
  #define MPI_INT ((MPI_Datatype) 42)
  // The number 42 is a type tag. It is tied to type "int".
  ```

The `type_tag_for_datatype` attribute also accepts an optional third argument
that determines how the type of the function argument specified by either
`arg_idx` or `ptr_idx` is compared against the type associated with the type
tag. (Recall that for the `argument_with_type_tag` attribute, the type of the
function argument specified by `arg_idx` is compared against the type
associated with the type tag. Also recall that for the `pointer_with_type_tag`
attribute, the pointee type of the function argument specified by `ptr_idx` is
compared against the type associated with the type tag.) There are two supported
values for this optional third argument:

- `layout_compatible` will cause types to be compared according to
  layout-compatibility rules (In C++11 [class.mem] p 17, 18, see the
  layout-compatibility rules for two standard-layout struct types and for two
  standard-layout union types). This is useful when creating a type tag
  associated with a struct or union type. For example:

  ```c++
  /* In mpi.h */
  typedef int MPI_Datatype;
  struct internal_mpi_double_int { double d; int i; };
  extern struct mpi_datatype mpi_datatype_double_int
      __attribute__(( type_tag_for_datatype(mpi,
                      struct internal_mpi_double_int, layout_compatible) ));

  #define MPI_DOUBLE_INT ((MPI_Datatype) &mpi_datatype_double_int)

  int MPI_Send(void *buf, int count, MPI_Datatype datatype, ...)
      __attribute__(( pointer_with_type_tag(mpi,1,3) ));

  /* In user code */
  struct my_pair { double a; int b; };
  struct my_pair *buffer;
  MPI_Send(buffer, 1, MPI_DOUBLE_INT /*, ... */); // no warning because the
                                                   // layout of my_pair is
                                                   // compatible with that of
                                                   // internal_mpi_double_int

  struct my_int_pair { int a; int b; }
  struct my_int_pair *buffer2;
  MPI_Send(buffer2, 1, MPI_DOUBLE_INT /*, ... */); // warning because the
                                                    // layout of my_int_pair
                                                    // does not match that of
                                                    // internal_mpi_double_int
  ```

- `must_be_null` specifies that the function argument specified by either
  `arg_idx` (for the `argument_with_type_tag` attribute) or `ptr_idx` (for
  the `pointer_with_type_tag` attribute) should be a null pointer constant.
  The second argument to the `type_tag_for_datatype` attribute is ignored. For
  example:

  ```c++
  /* In mpi.h */
  typedef int MPI_Datatype;
  extern struct mpi_datatype mpi_datatype_null
      __attribute__(( type_tag_for_datatype(mpi, void, must_be_null) ));

  #define MPI_DATATYPE_NULL ((MPI_Datatype) &mpi_datatype_null)
  int MPI_Send(void *buf, int count, MPI_Datatype datatype, ...)
      __attribute__(( pointer_with_type_tag(mpi,1,3) ));

  /* In user code */
  struct my_pair { double a; int b; };
  struct my_pair *buffer;
  MPI_Send(buffer, 1, MPI_DATATYPE_NULL /*, ... */); // warning: MPI_DATATYPE_NULL
                                                      // was specified but buffer
                                                      // is not a null pointer
  ```


