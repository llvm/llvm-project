# [Clang] Inconsistent big-endian layout for packed boolean vectors

On big-endian ARM, static initialization of a two-element boolean vector uses
the low bits of a byte, while loads and stores use the high bits.

## Reproducer

Save as `repro.c` (reproduced with Clang `24.0.0git`):

```c
typedef _Bool bool2 __attribute__((ext_vector_type(2)));

bool2 global_v = {1, 1};

void set(bool2 *p) {
  *p = (bool2){1, 1};
}

_Bool first(void) {
  return global_v[0];
}
```

```sh
clang --target=armv7-none-eabi -march=armv7-m -mbig-endian \
  -O1 -S repro.c -o -
```

**Actual:** `global_v` is emitted as `0x03`, but `set()` stores `0xC0`.
`first()` reads bit 7, returning `0` initially and `1` after `set(&global_v)`.

**Expected:** Initialization, loads, and stores must agree; `first()` should
return `1` in both cases. The little-endian variant behaves correctly.

[Original discussion](https://github.com/llvm/llvm-project/pull/224033#discussion_r4047046129)
and [Compiler Explorer example](https://godbolt.org/z/Y5ae6hjn9).
