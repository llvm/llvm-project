// UNSUPPORTED: target={{.*}}-aix{{.*}}

// Copying clang to a new location and running it will not work unless it is
// statically linked. Dynamically linked builds typically use relative rpaths,
// which this will break.
// REQUIRES: static-libs

// Test when clang is in a path containing a space.
// The initial `rm` is a workaround for https://openradar.appspot.com/FB8914243
// (Scenario: Run tests once, `clang` gets copied and run at new location and signature
// is cached at the new clang's inode, then clang is changed, tests run again, old signature
// is still cached with old clang's inode, so it won't execute this time. Removing the dir
// first guarantees a new inode without old cache entries.)
// RUN: rm -rf "%t.r/with spaces"
// RUN: mkdir -p "%t.r/with spaces"
// RUN: cp %clang "%t.r/with spaces/clang"
// RUN: "%t.r/with spaces/clang" -### -S --target=x86_64-unknown-linux -frecord-gcc-switches %s 2>&1 | FileCheck -check-prefix=CHECK-RECORD-GCC-SWITCHES-ESCAPED %s
// CHECK-RECORD-GCC-SWITCHES-ESCAPED: "-record-command-line" "{{.+}}with\\ spaces{{.+}}"
// Clean up copy of large binary copied into temp directory to avoid bloat.
// RUN: rm -f "%t.r/with spaces/clang" || true
