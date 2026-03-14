// RUN: %clang_cc1 -triple i386-pc-cygwin -E -x c %s
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -E -fwchar-type=short -fno-signed-wchar -x c %s

#if (L'\0' - 1 < 0)
# error "Unexpected expression evaluation result"
#endif
