// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

int dereference(int *x) {
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

int foobar(bool cond, int *x) {
  if (cond)
    x = 0;
  return dereference(x);
}

// CHECK:          <svg
// CHECK:            <g
// CHECK-COUNT-9:      <path class="arrow" id="arrow{{[0-9]+}}"/>
// CHECK-NOT:          <path class="arrow" id="arrow{{[0-9]+}}"/>
// CHECK:            </g>
// CHECK-NEXT:     </svg>
// CHECK-NEXT:     <script type='text/javascript'>
// CHECK-NEXT:     const arrowIndices = [ 9,8,6,5,3,2,0 ]
// CHECK-NEXT:     </script>
//
// Except for arrows we still want to have grey bubbles with control notes.
// CHECK:          <div id="Path2" class="msg msgControl"
// CHECK-SAME:       <div class="PathIndex PathIndexControl">2</div>
// CHECK-SAME:       <td>Taking true branch</td>
