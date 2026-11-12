/* The run lines are below, because this test is line- and
   column-number sensitive. */
void atAvailable() {
  if (@available(macOS 10.10, *)) {

  }
  if (__builtin_available(iOS 8, *)) {
  }
}

// RUN: c-index-test -code-completion-at=%s:4:18 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:7:27 %s | FileCheck %s
// CHECK: {TypedText iOS} (40)
// CHECK: {TypedText iOSApplicationExtension} (40)
// CHECK: {TypedText macOS} (40)
// CHECK: {TypedText macOSApplicationExtension} (40)
// CHECK: {TypedText tvOS} (40)
// CHECK: {TypedText tvOSApplicationExtension} (40)
// CHECK: {TypedText watchOS} (40)
// CHECK: {TypedText watchOSApplicationExtension} (40)
