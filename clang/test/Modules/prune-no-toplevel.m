// NetBSD: noatime mounts currently inhibit 'touch -a' updates
// UNSUPPORTED: system-netbsd

// Test that implicit module builds don't prune top-level files in the module
// cache directory.

// Set up a module cache with a timestamp old enough to trigger pruning, a
// top-level .pcm, and a stale .pcm in a subdirectory.
// RUN: rm -rf %t
// RUN: mkdir -p %t/cache/subdir
// RUN: touch -m -a -t 201101010000 %t/cache/modules.timestamp
// RUN: touch -a -t 201101010000 %t/cache/toplevel.pcm
// RUN: touch -a -t 201101010000 %t/cache/subdir/stale.pcm

// Run the compiler to trigger pruning.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify

// The top-level .pcm file should still exist.
// RUN: ls %t/cache/toplevel.pcm

// The subdirectory .pcm file should have been pruned.
// RUN: not ls %t/cache/subdir/stale.pcm

// expected-no-diagnostics
