// RUN: %clang_cc1 -analyze -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:Bind=true -analyzer-config debug.AnalysisOrder:RegionChanges=true %s 2>&1 | FileCheck %s

struct Super {
  virtual void m();
};
struct Sub : Super {
  virtual void m() {}
};

void testTemporaries() {
  // This triggers RegionChanges once for zero-initialization of the structure.
  Sub().m();
}

void seeIfCheckBindWorks() {
  // This should trigger checkBind. The rest of the code shouldn't.
  // This also triggers checkRegionChanges after that.
  // Note that this function is analyzed first, so the messages would be on top.
  int x = 1;
}

// seeIfCheckBindWorks():
// CHECK: Bind
// CHECK-NEXT: RegionChanges

// testTemporaries():
// CHECK-NEXT: RegionChanges

// Make sure there's no further output.
// CHECK-NOT: Bind
// CHECK-NOT: RegionChanges
