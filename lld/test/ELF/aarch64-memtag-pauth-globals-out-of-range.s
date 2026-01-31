# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android %s -o %t.o
# RUN: not ld.lld --shared --android-memtag-mode=sync %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --implicit-check-not=error:

## Verify that, when composing PAuth and Memtag ABIs, we error if trying to
## emit an R_AARCH64_AUTH_RELATIVE for a tagged global using an original addend
## that's not within the bounds of the symbol and, when negated, does not fit
## in a signed 32-bit integer, since the signing schema uses the upper 32 bits.

# CHECK: error: {{.*}}aarch64-memtag-pauth-globals-out-of-range.s.tmp.o:(.data+0x0): relocation R_AARCH64_AUTH_ABS64 out of range: 2147483648 is not in [-2147483648, 2147483647]; references 'foo'
# CHECK: error: {{.*}}aarch64-memtag-pauth-globals-out-of-range.s.tmp.o:(.data+0x8): relocation R_AARCH64_AUTH_ABS64 out of range: -2147483649 is not in [-2147483648, 2147483647]; references 'foo'

.data
.balign 16
.memtag foo
.type foo, @object
foo:
.quad (foo - 0x80000000)@AUTH(da,42)
.quad (foo + 0x80000001)@AUTH(da,42)
## These are just in bounds
.quad (foo - 0x7fffffff)@AUTH(da,42)
.quad (foo + 0x80000000)@AUTH(da,42)
.size foo, 32
