


typedef int T;
void OBSCURE(func)(int x) {
  OBSCURE(T) DECORATION value;
}


// Without PCH
// RUN: c-index-test -cursor-at=%s.h:1:11 \
// RUN:              -cursor-at=%s.h:2:14 \
// RUN:              -cursor-at=%s.h:4:5 \
// RUN:              -cursor-at=%s.h:5:5 \
// RUN:              -cursor-at=%s.h:5:14 \
// RUN:              -cursor-at=%s:5:7 \
// RUN:              -cursor-at=%s:6:6 \
// RUN:              -cursor-at=%s:6:19 \
// RUN:     -include %s.h %s | FileCheck %s

// With PCH
// RUN: c-index-test -write-pch %t.h.pch %s.h -Xclang -detailed-preprocessing-record
// RUN: c-index-test -cursor-at=%s.h:1:11 \
// RUN:              -cursor-at=%s.h:2:14 \
// RUN:              -cursor-at=%s.h:4:5 \
// RUN:              -cursor-at=%s.h:5:5 \
// RUN:              -cursor-at=%s.h:5:14 \
// RUN:              -cursor-at=%s:5:7 \
// RUN:              -cursor-at=%s:6:6 \
// RUN:              -cursor-at=%s:6:19 \
// RUN:     -include %t.h %s | FileCheck %s

// From header
// CHECK: macro definition=OBSCURE
// CHECK: macro definition=DECORATION
// CHECK: macro expansion=DECORATION:2:9
// CHECK: macro expansion=OBSCURE:1:9
// CHECK: macro expansion=DECORATION:2:9

// From main file
// CHECK: macro expansion=OBSCURE:1:9
// CHECK: macro expansion=OBSCURE:1:9
// CHECK: macro expansion=DECORATION:2:9
