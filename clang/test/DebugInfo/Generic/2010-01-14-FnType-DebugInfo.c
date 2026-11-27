// RUN: %clang_cc1 %s -emit-llvm -debug-info-kind=limited -o /dev/null
typedef void (*sigcatch_t)( struct sigcontext *);
sigcatch_t sigcatch[50] = {(sigcatch_t) 0};

