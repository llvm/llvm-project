# Reporting LLVM Security Issues

To report security issues in LLVM, please follow the steps outlined on the
[LLVM Security Group](https://llvm.org/docs/Security.html#how-to-report-a-security-issue)
page.

## Security Issue Scope

Many of LLVM's tools are explicitly **not** considered to be hardened against
malicious input. Bugs in LLVM tools like buffer overreads or crashes are
valuable to report [as Issues](https://github.com/llvm/llvm-project/issues),
but aren't always seen as security vulnerabilities. Please see
[our documentation](https://llvm.org/docs/Security.html#what-is-considered-a-security-issue)
for a more precise definition of a security issue in this repository.
