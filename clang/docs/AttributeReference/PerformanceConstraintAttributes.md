## Performance Constraint Attributes

The `nonblocking`, `blocking`, `nonallocating` and `allocating` attributes can be attached
to function types, including blocks, C++ lambdas, and member functions. The attributes declare
constraints about a function's behavior pertaining to blocking and heap memory allocation.

There are several rules for function types with these attributes, enforced with
compiler warnings:

- When assigning or otherwise converting to a function pointer of `nonblocking` or
  `nonallocating` type, the source must also be a function or function pointer of
  that type, unless it is a null pointer, i.e. the attributes should not be "spoofed". Conversions
  that remove the attributes are transparent and valid.
- An override of a `nonblocking` or `nonallocating` virtual method must also be declared
  with that same attribute (or a stronger one.) An overriding method may add an attribute.
- A redeclaration of a `nonblocking` or `nonallocating` function must also be declared with
  the same attribute (or a stronger one). A redeclaration may add an attribute.

The warnings are controlled by `-Wfunction-effects`, which is disabled by default.

The compiler also diagnoses function calls from `nonblocking` and `nonallocating`
functions to other functions which lack the appropriate attribute.

### allocating

{clang-attr-syntaxes}`AllocatingDocs`

Declares that a function potentially allocates heap memory, and prevents any potential inference
of `nonallocating` by the compiler.


### blocking

{clang-attr-syntaxes}`BlockingDocs`

Declares that a function potentially blocks, and prevents any potential inference of `nonblocking`
by the compiler.


### nonallocating

{clang-attr-syntaxes}`NonAllocatingDocs`

Declares that a function or function type either does or does not allocate heap memory, according
to the optional, compile-time constant boolean argument, which defaults to true. When the argument
is false, the attribute is equivalent to `allocating`.


### nonblocking

{clang-attr-syntaxes}`NonBlockingDocs`

Declares that a function or function type either does or does not block in any way, according
to the optional, compile-time constant boolean argument, which defaults to true. When the argument
is false, the attribute is equivalent to `blocking`.

For the purposes of diagnostics, `nonblocking` is considered to include the
`nonallocating` guarantee and is therefore a "stronger" constraint or attribute.


