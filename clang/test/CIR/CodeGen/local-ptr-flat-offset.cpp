// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o %t.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s

template <class Iter> struct reverse_iterator {
  Iter t;
};

struct S {
  int x;
};

S gS;

void use(reverse_iterator<S *> &);

void test() {
  reverse_iterator<S *> it{&gS + 1};
  use(it);
}

// CIR: cir.global "private" constant cir_private @__const._Z4testv.it =
// CIR-SAME: #cir.const_record<{#cir.global_offset<@gS, 4> : !cir.ptr<!rec_S>}>

// CIR-LABEL: cir.func {{.*}} @_Z4testv()
// CIR:         %[[IT:.*]] = cir.alloca "it" {{.*}} !cir.ptr<!rec_reverse_iterator{{.*}}>
// CIR:         %[[CONST:.*]] = cir.get_global @__const._Z4testv.it
// CIR:         cir.copy %[[CONST]] to %[[IT]]

// LLVM: @__const._Z4testv.it = {{.*}}constant {{.*}}{ ptr getelementptr {{.*}}(i8, ptr @gS, i64 4) }

// LLVM-LABEL: define {{.*}} @_Z4testv()
// LLVM:         %[[IT:.*]] = alloca %{{.*}}reverse_iterator
// LLVM:         call void @llvm.memcpy{{.*}}(ptr {{.*}}%[[IT]], ptr {{.*}}@__const._Z4testv.it
