

moving all calls to syscall_impl into OSUtil/linux/syscall_wrappers/

Form:
```
LIBC_INLINE ErrorOr<return_type> function_name(args) {
  return_type ret = syscall_impl<return_type>(SYS_function_name, args);
  if (ret < 0) {
    return Error(-static_cast<int>(ret));
  }
  return Ok(ret);
}
```

Update all other calls.
