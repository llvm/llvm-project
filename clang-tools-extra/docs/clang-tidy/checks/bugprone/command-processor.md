```{title} clang-tidy - bugprone-command-processor
```

# bugprone-command-processor

Flags calls to `system()`, `popen()`, and `_popen()`, which
execute a command processor. It does not flag calls to `system()` with a null
pointer argument, as such a call checks for the presence of a command processor
but does not actually attempt to execute a command.

## References

This check corresponds to the CERT C Coding Standard rule
[ENV33-C. Do not call system()][ENV33-C].

[ENV33-C]: https://cmu-sei.github.io/secure-coding-standards/sei-cert-c-coding-standard/rules/environment-env/env33-c/
