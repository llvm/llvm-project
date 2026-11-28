// Builtins are only stored in a PCH if it uses them. Check that the importer
// can use builtins the PCH doesn't know about, including target builtins, and
// that it applies its own -fno-builtin-* to those.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple arm64-apple-macosx -target-feature +neon \
// RUN:   -emit-pch -o %t/prefix.pch %t/prefix.h
// RUN: %clang_cc1 -triple arm64-apple-macosx -target-feature +neon \
// RUN:   -include-pch %t/prefix.pch -fsyntax-only -verify %t/tu.c
// RUN: %clang_cc1 -triple arm64-apple-macosx -target-feature +neon \
// RUN:   -include-pch %t/prefix.pch -fsyntax-only -verify=no-strlen \
// RUN:   -fno-builtin-strlen %t/tu.c

// Same with chained PCHs.
// RUN: %clang_cc1 -triple arm64-apple-macosx -target-feature +neon \
// RUN:   -chain-include %t/prefix.h -chain-include %t/chained.h \
// RUN:   -fsyntax-only -verify -DCHAINED %t/tu.c

//--- prefix.h
int uses_one_builtin(unsigned x) { return __builtin_popcount(x); }

//--- chained.h
_Static_assert(__builtin_ctz(8) == 3, "");
#if !__has_builtin(__builtin_arm_rbit)
#error "missing target builtin"
#endif

//--- tu.c
// expected-no-diagnostics
_Static_assert(__builtin_popcount(7) == 3, "");
_Static_assert(__builtin_clz(1) == 31, "");

#if !__has_builtin(__builtin_neon_vaddq_v) || !__has_builtin(__builtin_arm_rbit)
#error "missing target builtins"
#endif

#if !__has_builtin(__builtin_strlen)
#error "missing __builtin_strlen"
#endif

unsigned long strlen(const char *);
// no-strlen-error@+1 {{static assertion expression is not an integral constant expression}}
_Static_assert(strlen("abc") == 3, "");
