// RUN: %clang_cc1 -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#define M if (x) goto L1;
#define M10 M M M M M M M M M M
#define M100 M10 M10 M10 M10 M10 M10 M10 M10 M10 M10
#define M1000 M100 M100 M100 M100 M100 M100 M100 M100 M100 M100

void f(int x) {
  int h;

  // Many gotos to not-yet-emitted labels would cause EHScope's FixupDepth
  // to overflow (PR23490).
  M1000 M1000 M1000

  if (x == 5) {
    // This will cause us to emit a clean-up of the stack variable. If the
    // FixupDepths are broken, fixups will erroneously get threaded through it.
    int i;
  }

L1:
  return;
}

// CHECK-LABEL: define {{(dso_local )?}}void @f
// CHECK-NOT: cleanup
// CHECK: {{^!0 = }}
// We might see 'cleanup' as part of a pathname or versioning in metadata
