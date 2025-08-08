// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -mno-constructor-aliases %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -mno-constructor-aliases -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -mno-constructor-aliases -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void some_function() noexcept;

struct out_of_line_destructor {
    int prevent_tail_padding_reuse;
    ~out_of_line_destructor();
};

out_of_line_destructor::~out_of_line_destructor() {
    some_function();
}

// CIR: !rec_out_of_line_destructor = !cir.record<struct "out_of_line_destructor" {!s32i}>

// CIR: cir.func dso_local @_ZN22out_of_line_destructorD2Ev(%{{.+}}: !cir.ptr<!rec_out_of_line_destructor>
// CIR:   cir.call @_Z13some_functionv() nothrow : () -> () 
// CIR:   cir.return 

// LLVM: define dso_local void @_ZN22out_of_line_destructorD2Ev(ptr %{{.+}})
// LLVM:   call void @_Z13some_functionv()
// LLVM:   ret void

// OGCG: define dso_local void @_ZN22out_of_line_destructorD2Ev(ptr {{.*}}%{{.+}})
// OGCG:   call void @_Z13some_functionv()
// OGCG:   ret void

// CIR: cir.func dso_local @_ZN22out_of_line_destructorD1Ev(%{{.+}}: !cir.ptr<!rec_out_of_line_destructor>
// CIR:  cir.call @_ZN22out_of_line_destructorD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_out_of_line_destructor>)
// CIR:  cir.return

// LLVM: define dso_local void @_ZN22out_of_line_destructorD1Ev(ptr %{{.+}})
// LLVM:   call void @_ZN22out_of_line_destructorD2Ev
// LLVM:   ret void

// OGCG: define dso_local void @_ZN22out_of_line_destructorD1Ev(ptr {{.*}}%{{.+}})
// OGCG:   call void @_ZN22out_of_line_destructorD2Ev
// OGCG:   ret void

struct inline_destructor {
    int prevent_tail_padding_reuse;
    ~inline_destructor() noexcept(false) {
        some_function();
    }
};

// This inline destructor is not odr-used in this TU.
// Make sure we don't emit a definition

// CIR-NOT: cir.func {{.*}}inline_destructor{{.*}}
// LLVM-NOT: define {{.*}}inline_destructor{{.*}}
// OGCG-NOT: define {{.*}}inline_destructor{{.*}}
