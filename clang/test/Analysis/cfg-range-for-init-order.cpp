// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=false %s 2>&1 | FileCheck %s
// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=debug.DumpCFG -analyzer-config cfg-rich-constructors=true %s 2>&1 | FileCheck %s

struct Begin {
  ~Begin();
};

struct End {
  ~End();
};

bool operator!=(const Begin &, const End &);
void operator++(Begin &);
int &operator*(const Begin &);

struct Range {
  Begin begin();
  End end();
};

// The cleanup block appears before the initialization block in the dump.
// Iterators must initialize begin before end and be destroyed in reverse order.
//
// CHECK-LABEL: void range_for_init_order(
// CHECK:      [B1]
// CHECK-NEXT:   1: [B5.12].~End() (Implicit destructor)
// CHECK-NEXT:   2: [B5.7].~Begin() (Implicit destructor)
// CHECK-NEXT:   Preds (1): B2
// CHECK-NEXT:   Succs (1): B0
// CHECK-EMPTY:
//
// CHECK:      [B5]
// CHECK-NEXT:   1: range
// CHECK-NEXT:   2: auto &&__range1 = range;
// CHECK-NEXT:   3: __range1
// CHECK-NEXT:   4: [B5.3].begin
// CHECK-NEXT:   5: [B5.4](){{.*}}
// CHECK-NEXT:   6: [B5.5] (BindTemporary)
// CHECK-NEXT:   7: auto __begin1 = __range1.begin();
// CHECK-NEXT:   8: __range1
// CHECK-NEXT:   9: [B5.8].end
// CHECK-NEXT:  10: [B5.9](){{.*}}
// CHECK-NEXT:  11: [B5.10] (BindTemporary)
// CHECK-NEXT:  12: auto __end1 = __range1.end();
// CHECK-NEXT:   Preds (1): B6
// CHECK-NEXT:   Succs (1): B2
// CHECK-EMPTY:
void range_for_init_order(Range range) {
  for (int &value : range)
    (void)value;
}
