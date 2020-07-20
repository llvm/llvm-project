// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

void bar(int);

void foo() {
  int a;
  bar(a); // expected-warning{{1st function call argument is an uninitialized value}}
}

// CHECK-LABEL:    <div id="EndPath" class="msg msgEvent" style="margin-left:3ex">
// CHECK-SAME:       <table class="msgT">
// CHECK-SAME:         <tr>
// CHECK-SAME:           <td valign="top">
// CHECK-SAME:             <div class="PathIndex PathIndexEvent">2</div>
// CHECK-SAME:           </td>
// CHECK-SAME:           <td>
// CHECK-SAME:             <div class="PathNav">
// CHECK-SAME:               <a href="#Path1" title="Previous event (1)">&#x2190;</a>
// CHECK-SAME:             </div>
// CHECK-SAME:           </td>
// CHECK-NOT:            </td>
// CHECK-SAME:           <td>
// CHECK-SAME:             1st function call argument is an uninitialized value
// CHECK-SAME:           </td>
// CHECK-SAME:         </tr>
// CHECK-SAME:       </table>
// CHECK-SAME:     </div>
