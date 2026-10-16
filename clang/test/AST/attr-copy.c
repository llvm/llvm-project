// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-deprecated-declarations -verify -ast-dump %s | FileCheck %s

void *source(int) __attribute__((malloc, alloc_size(1), nothrow, returns_nonnull,
                                deprecated, weak, visibility("hidden")));

// CHECK-LABEL: FunctionDecl {{.*}} copied 'void *(int)'
// CHECK: RestrictAttr
// CHECK: AllocSizeAttr {{.*}} 1
// CHECK: NoThrowAttr
// CHECK: ReturnsNonNullAttr
// CHECK-NOT: DeprecatedAttr
// CHECK-NOT: WeakAttr
// CHECK-NOT: VisibilityAttr
// CHECK-NOT: CopyAttr
void *copied(int) __attribute__((copy(source)));

// CHECK-LABEL: FunctionDecl {{.*}} chain 'void *(int)'
// CHECK: RestrictAttr
// CHECK: AllocSizeAttr {{.*}} 1
// CHECK: NoThrowAttr
// CHECK: ReturnsNonNullAttr
void *chain(int) __attribute__((copy(copied)));

void target(void) {}
void alias_source(void) __attribute__((alias("target"), cold));
// CHECK-LABEL: FunctionDecl {{.*}} alias_copy 'void (void)'
// CHECK-NEXT: ColdAttr
// CHECK-NOT: AliasAttr
void alias_copy(void) __attribute__((copy(alias_source)));

void multiversion(void) __attribute__((target_clones("default", "sse4.2"), nothrow));
// CHECK-LABEL: FunctionDecl {{.*}} multiversion_copy 'void (void)'
// CHECK-NEXT: NoThrowAttr
// CHECK-NOT: TargetClonesAttr
void multiversion_copy(void) __attribute__((copy(multiversion)));

int source_var __attribute__((aligned(32), section("data"), weak));
// CHECK-LABEL: VarDecl {{.*}} copied_var 'int'
// CHECK: AlignedAttr
// CHECK: IntegerLiteral {{.*}} 32
// CHECK: SectionAttr {{.*}} "data"
// CHECK-NOT: WeakAttr
int copied_var __attribute__((copy(&source_var)));

void deallocate(void *);
// expected-warning@+1 {{'malloc' attribute ignored because Clang does not yet support this attribute signature}}
void *allocate(int, int) __attribute__((malloc(deallocate, 1), alloc_size(1, 2),
                                      alloc_align(1), assume_aligned(32, 4)));
// CHECK-LABEL: FunctionDecl {{.*}} allocate_copy 'void *(int, int)'
// CHECK: RestrictAttr
// CHECK: AllocSizeAttr {{.*}} 1 2
// CHECK: AllocAlignAttr {{.*}} 1
// CHECK: AssumeAlignedAttr
void *allocate_copy(int, int) __attribute__((copy(allocate))); // expected-warning {{'malloc' attribute ignored because Clang does not yet support this attribute signature}}

void zero_source(void) __attribute__((zero_call_used_regs("used")));
// CHECK-LABEL: FunctionDecl {{.*}} zero_copy 'void (void)'
// CHECK-NEXT: ZeroCallUsedRegsAttr {{.*}} Used
void zero_copy(void) __attribute__((copy(zero_source)));

void cleanup(int *);
void local(void) {
  int source_local __attribute__((cleanup(cleanup)));
  // CHECK: VarDecl {{.*}} copy_local 'int'
  // CHECK-NEXT: CleanupAttr {{.*}} 'cleanup'
  int copy_local __attribute__((copy(source_local)));
}

void always_inline_source(void) __attribute__((always_inline));
void noinline_source(void) __attribute__((noinline));
inline void gnu_inline_source(void) __attribute__((gnu_inline));

// Inlining attributes are excluded from copy even when the destination has no
// conflicting attribute. This is not an explicit-attribute precedence rule.
// CHECK-LABEL: FunctionDecl {{.*}} plain_always_inline_copy 'void (void)'
// CHECK-NOT: Attr
void plain_always_inline_copy(void) __attribute__((copy(always_inline_source)));
// CHECK-LABEL: FunctionDecl {{.*}} plain_noinline_copy 'void (void)'
// CHECK-NOT: Attr
void plain_noinline_copy(void) __attribute__((copy(noinline_source)));
// CHECK-LABEL: FunctionDecl {{.*}} plain_gnu_inline_copy 'void (void)'
// CHECK-NOT: Attr
void plain_gnu_inline_copy(void) __attribute__((copy(gnu_inline_source)));

// CHECK-LABEL: FunctionDecl {{.*}} explicit_noinline_first 'void (void)'
// CHECK-NEXT: NoInlineAttr
// CHECK-NOT: Attr
void explicit_noinline_first(void)
    __attribute__((noinline, copy(always_inline_source)));
// CHECK-LABEL: FunctionDecl {{.*}} explicit_noinline_last 'void (void)'
// CHECK-NEXT: NoInlineAttr
// CHECK-NOT: Attr
void explicit_noinline_last(void)
    __attribute__((copy(always_inline_source), noinline));

// CHECK-LABEL: FunctionDecl {{.*}} explicit_always_inline_first 'void (void)'
// CHECK-NEXT: AlwaysInlineAttr
// CHECK-NOT: Attr
void explicit_always_inline_first(void)
    __attribute__((always_inline, copy(noinline_source)));
// CHECK-LABEL: FunctionDecl {{.*}} explicit_always_inline_last 'void (void)'
// CHECK-NEXT: AlwaysInlineAttr
// CHECK-NOT: Attr
void explicit_always_inline_last(void)
    __attribute__((copy(noinline_source), always_inline));
