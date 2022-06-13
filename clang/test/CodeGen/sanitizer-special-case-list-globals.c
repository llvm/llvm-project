/// Verify that ignorelist sections correctly select sanitizers to apply
/// ignorelist entries to.

// RUN: %clang_cc1 -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=NONE

// RUN: %clang_cc1 -fsanitize=address -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=ASAN

/// Note: HWASan effectively reorders globals (it puts the unsanitized ones
/// first), which is hard to check for, as 'CHECK-DAG' doesn't play terribly
/// nicely with 'CHECK-NOT'. This is why the 'always_ignored' and
/// 'hwasan_ignored' comes first in this file.
// RUN: %clang_cc1 -fsanitize=hwaddress -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=HWASAN

/// TODO(hctim): Move over to memtag-globals when it's implemented. For now
/// though, it's fine, the frontend still annotates based on any memtag sanitizer
/// being used.
// RUN: %clang_cc1 -fsanitize=memtag-heap -triple=aarch64-linux-android31 -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=MEMTAG

/// Check that the '[cfi-vcall|cfi-icall] src:*' rule in the ignorelist doesn't change
/// anything for ASan.
// RUN: %clang_cc1 -fsanitize=address -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=ASAN

/// Check that -fsanitize=kernel-address picks up the '[address]' groups.
// RUN: %clang_cc1 -fsanitize=kernel-address -mllvm -hwasan-kernel=1 -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=ASAN

/// KHWASan doesn't instrument global variables.
// RUN: %clang_cc1 -fsanitize=kernel-hwaddress -mllvm -hwasan-kernel=1 -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=NONE

/// Check that the '[cfi-vcall|cfi-icall] src:*' rule doesnt' emit anything for
/// GVs.
// RUN: %clang_cc1 -fsanitize=cfi-vcall,cfi-icall -emit-llvm %s -o -\
// RUN: -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list-globals.txt \
// RUN: | FileCheck %s --check-prefix=NONE

// NONE:     @always_ignored = global
// NONE-NOT: no_sanitize
// ASAN:     @always_ignored = global {{.*}}, no_sanitize_address
// HWASAN:   @always_ignored = global {{.*}}, no_sanitize_hwaddress
// MEMTAG:   @always_ignored = global {{.*}}, no_sanitize_memtag
unsigned always_ignored;

// NONE:       @hwasan_ignored = global
// NONE-NOT:   no_sanitize
// ASAN:       @hwasan_ignored = global
// ASAN-NOT:   no_sanitize_address
// HWASAN:     @hwasan_ignored = global {{.*}}, no_sanitize_hwaddress
// MEMTAG:     @hwasan_ignored = global
// MEMTAG-NOT: no_sanitize_memtag
unsigned hwasan_ignored;

// NONE:       @asan_ignored = global
// NONE-NOT:   asan_ignored
// ASAN:       @asan_ignored = global {{.*}}, no_sanitize_address
// HWASAN:     @asan_ignored.hwasan = {{.*}} global
// HWASAN-NOT: no_sanitize_hwaddress
// MEMTAG:     @asan_ignored = global
// MEMTAG-NOT: no_sanitize_memtag
unsigned asan_ignored;

// NONE:       @memtag_ignored = global
// NONE-NOT:   memtag_ignored
// ASAN:       @memtag_ignored = global
// ASAN-NOT:   no_sanitize_address
// HWASAN:     @memtag_ignored.hwasan = {{.*}} global
// HWASAN-NOT: no_sanitize_hwaddress
// MEMTAG:     @memtag_ignored = global {{.*}}, no_sanitize_memtag
unsigned memtag_ignored;

// NONE:       @never_ignored = global
// NONE-NOT:   never_ignored
// ASAN:       @never_ignored = global
// ASAN-NOT:   no_sanitize_address
// HWASAN:     @never_ignored.hwasan = {{.*}} global
// HWASAN-NOT: no_sanitize_hwaddress
// MEMTAG:     @never_ignored = global
// MEMTAG-NOT: no_sanitize_memtag
unsigned never_ignored;
