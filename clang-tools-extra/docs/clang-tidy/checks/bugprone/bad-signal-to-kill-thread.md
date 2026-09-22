```{title} clang-tidy - bugprone-bad-signal-to-kill-thread
```

# bugprone-bad-signal-to-kill-thread

Finds `pthread_kill` function calls when a thread is terminated by
raising `SIGTERM` signal and the signal kills the entire process, not
just the individual thread. Use any signal except `SIGTERM`.

```cpp
pthread_kill(thread, SIGTERM);
```

This check corresponds to the CERT C Coding Standard rule
[POS44-C. Do not use signals to terminate threads](https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/posix-pos/pos44-c/).

`cert-pos44-c` redirects here as an alias of this check.
