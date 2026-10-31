## Handle Attributes

Handles are a way to identify resources like files, sockets, and processes.
They are more opaque than pointers and widely used in system programming. They
have similar risks such as never releasing a resource associated with a handle,
attempting to use a handle that was already released, or trying to release a
handle twice. Using the annotations below it is possible to make the ownership
of the handles clear: whose responsibility is to release them. They can also
aid static analysis tools to find bugs.

### acquire_handle

{clang-attr-syntaxes}`AcquireHandleDocs`

If this annotation is on a function or a function type it is assumed to return
a new handle. In case this annotation is on an output parameter,
the function is assumed to fill the corresponding argument with a new
handle. The attribute requires a string literal argument which used to
identify the handle with later uses of `use_handle` or
`release_handle`.

```c++
// Output arguments from Zircon.
zx_status_t zx_socket_create(uint32_t options,
                             zx_handle_t __attribute__((acquire_handle("zircon"))) * out0,
                             zx_handle_t* out1 [[clang::acquire_handle("zircon")]]);


// Returned handle.
[[clang::acquire_handle("tag")]] int open(const char *path, int oflag, ... );
int open(const char *path, int oflag, ... ) __attribute__((acquire_handle("tag")));
```


### release_handle

{clang-attr-syntaxes}`ReleaseHandleDocs`

If a function parameter is annotated with `release_handle(tag)` it is assumed to
close the handle. It is also assumed to require an open handle to work with. The
attribute requires a string literal argument to identify the handle being released.

```c++
zx_status_t zx_handle_close(zx_handle_t handle [[clang::release_handle("tag")]]);
```


### use_handle

{clang-attr-syntaxes}`UseHandleDocs`

A function taking a handle by value might close the handle. If a function
parameter is annotated with `use_handle(tag)` it is assumed to not to change
the state of the handle. It is also assumed to require an open handle to work with.
The attribute requires a string literal argument to identify the handle being used.

```c++
zx_status_t zx_port_wait(zx_handle_t handle [[clang::use_handle("zircon")]],
                         zx_time_t deadline,
                         zx_port_packet_t* packet);
```


