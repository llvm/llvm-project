// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -E -x c %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -E -fno-signed-char -x c %s

#if (L'\0' - 1 > 0)
# error "Unexpected expression evaluation result"
#endif
