// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  ~Struk();
};

// CHECK: !rec_Struk = !cir.record<struct "Struk" padded {!u8i}>

// CHECK: cir.func{{.*}} @_ZN5StrukD1Ev(!cir.ptr<!rec_Struk>)

void test_cleanup() {
  Struk s;
}

// CHECK: cir.func{{.*}} @_Z12test_cleanupv()
// CHECK:   %[[S_ADDR:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s"]
// CHECK:   cir.call @_ZN5StrukD1Ev(%[[S_ADDR]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:   cir.return

void test_cleanup_ifelse(bool b) {
  if (b) {
    Struk s;
  } else {
    Struk s;
  }
}

// CHECK: cir.func{{.*}} @_Z19test_cleanup_ifelseb(%arg0: !cir.bool
// CHECK:   cir.scope {
// CHECK:     %[[B:.*]] = cir.load{{.*}} %0 : !cir.ptr<!cir.bool>
// CHECK:     cir.if %[[B]] {
// CHECK:       %[[S:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s"]
// CHECK:       cir.call @_ZN5StrukD1Ev(%[[S]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:     } else {
// CHECK:       %[[S_TOO:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s"]
// CHECK:       cir.call @_ZN5StrukD1Ev(%[[S_TOO]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return

void test_cleanup_for() {
  for (int i = 0; i < 10; i++) {
    Struk s;
  }
}

// CHECK: cir.func{{.*}} @_Z16test_cleanup_forv()
// CHECK:   cir.scope {
// CHECK:     cir.for : cond {
// CHECK:     } body {
// CHECK:       cir.scope {
// CHECK:         %[[S:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["s"]
// CHECK:         cir.call @_ZN5StrukD1Ev(%[[S]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:       }
// CHECK:       cir.yield
// CHECK:     } step {
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return

void test_cleanup_nested() {
  Struk outer;
  {
    Struk middle;
    {
      Struk inner;
    }
  }
}

// CHECK: cir.func{{.*}} @_Z19test_cleanup_nestedv()
// CHECK:   %[[OUTER:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["outer"]
// CHECK:   cir.scope {
// CHECK:     %[[MIDDLE:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["middle"]
// CHECK:     cir.scope {
// CHECK:       %[[INNER:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>, ["inner"]
// CHECK:       cir.call @_ZN5StrukD1Ev(%[[INNER]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:     }
// CHECK:     cir.call @_ZN5StrukD1Ev(%[[MIDDLE]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:   }
// CHECK:   cir.call @_ZN5StrukD1Ev(%[[OUTER]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:   cir.return

void use_ref(const Struk &);

void test_expr_with_cleanup() {
  use_ref(Struk{});
}

// CHECK: cir.func{{.*}} @_Z22test_expr_with_cleanupv()
// CHECK:   cir.scope {
// CHECK:     %[[S:.*]] = cir.alloca !rec_Struk, !cir.ptr<!rec_Struk>
// CHECK:     cir.call @_Z7use_refRK5Struk(%[[S]])
// CHECK:     cir.call @_ZN5StrukD1Ev(%[[S]]) nothrow : (!cir.ptr<!rec_Struk>) -> ()
// CHECK:   }
// CHECK:   cir.return
