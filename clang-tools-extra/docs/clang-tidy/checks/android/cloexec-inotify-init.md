```{title} clang-tidy - android-cloexec-inotify-init
```

# android-cloexec-inotify-init

The usage of `inotify_init()` is not recommended, it's better to use
`inotify_init1()`.

Examples:

```cpp
inotify_init();

// becomes

inotify_init1(IN_CLOEXEC);
```
