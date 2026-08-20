# The Darwin sigsetjmp epilogue is disabled: the upstream implementation
# (sigsetjmp_epilogue.cpp) is incomplete on Darwin - it relies on a
# "sigprocmask" syscall and on __jmp_buf.sigmask, neither of which exists on
# Darwin (there is no libc/src/signal/darwin/ implementation either). Building
# it fails, so sigsetjmp/siglongjmp are not provided on Darwin until upstream
# implements the Darwin signal-mask machinery.
