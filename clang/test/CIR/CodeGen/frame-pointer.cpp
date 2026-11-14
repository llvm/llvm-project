// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-default.cir
// RUN: FileCheck --input-file=%t-default.cir %s --check-prefix=CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-default-cir.ll
// RUN: FileCheck --input-file=%t-default-cir.ll %s --check-prefix=LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-default.ll
// RUN: FileCheck --input-file=%t-default.ll %s --check-prefix=LLVM-NONE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=none -fclangir -emit-cir %s -o %t-none.cir
// RUN: FileCheck --input-file=%t-none.cir %s --check-prefix=CIR-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=none -fclangir -emit-llvm %s -o %t-none-cir.ll
// RUN: FileCheck --input-file=%t-none-cir.ll %s --check-prefix=LLVM-NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=none -emit-llvm %s -o %t-none.ll
// RUN: FileCheck --input-file=%t-none.ll %s --check-prefix=LLVM-NONE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf -fclangir -emit-cir %s -o %t-non-leaf.cir
// RUN: FileCheck --input-file=%t-non-leaf.cir %s --check-prefix=CIR-NON-LEAF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf -fclangir -emit-llvm %s -o %t-non-leaf-cir.ll
// RUN: FileCheck --input-file=%t-non-leaf-cir.ll %s --check-prefix=LLVM-NON-LEAF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf -emit-llvm %s -o %t-non-leaf.ll
// RUN: FileCheck --input-file=%t-non-leaf.ll %s --check-prefix=LLVM-NON-LEAF

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=all -fclangir -emit-cir %s -o %t-all.cir
// RUN: FileCheck --input-file=%t-all.cir %s --check-prefix=CIR-ALL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=all -fclangir -emit-llvm %s -o %t-all-cir.ll
// RUN: FileCheck --input-file=%t-all-cir.ll %s --check-prefix=LLVM-ALL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=all -emit-llvm %s -o %t-all.ll
// RUN: FileCheck --input-file=%t-all.ll %s --check-prefix=LLVM-ALL

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=reserved -fclangir -emit-cir %s -o %t-reserved.cir
// RUN: FileCheck --input-file=%t-reserved.cir %s --check-prefix=CIR-RESERVED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=reserved -fclangir -emit-llvm %s -o %t-reserved-cir.ll
// RUN: FileCheck --input-file=%t-reserved-cir.ll %s --check-prefix=LLVM-RESERVED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=reserved -emit-llvm %s -o %t-reserved.ll
// RUN: FileCheck --input-file=%t-reserved.ll %s --check-prefix=LLVM-RESERVED

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf-no-reserve -fclangir -emit-cir %s -o %t-nlnr.cir
// RUN: FileCheck --input-file=%t-nlnr.cir %s --check-prefix=CIR-NLNR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf-no-reserve -fclangir -emit-llvm %s -o %t-nlnr-cir.ll
// RUN: FileCheck --input-file=%t-nlnr-cir.ll %s --check-prefix=LLVM-NLNR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mframe-pointer=non-leaf-no-reserve -emit-llvm %s -o %t-nlnr.ll
// RUN: FileCheck --input-file=%t-nlnr.ll %s --check-prefix=LLVM-NLNR

void declaration();
void definition() { declaration(); }

// CIR-NONE-NOT: frame_pointer
// CIR-NONE: cir.func {{.*}}@_Z10definitionv()
// CIR-NONE-NOT: frame_pointer

// CIR-NON-LEAF: module {{.*}}attributes {{.*}}cir.frame_pointer = #cir.frame_pointer<non_leaf>
// CIR-NON-LEAF: cir.func {{.*}}@_Z10definitionv() attributes {{.*}}frame_pointer = #cir.frame_pointer<non_leaf>
// CIR-NON-LEAF: cir.func private @_Z11declarationv() attributes {{.*}}frame_pointer = #cir.frame_pointer<non_leaf>

// CIR-ALL: module {{.*}}attributes {{.*}}cir.frame_pointer = #cir.frame_pointer<all>
// CIR-ALL: cir.func {{.*}}@_Z10definitionv() attributes {{.*}}frame_pointer = #cir.frame_pointer<all>
// CIR-ALL: cir.call @_Z11declarationv() : () -> () loc(
// CIR-ALL: cir.func private @_Z11declarationv() attributes {{.*}}frame_pointer = #cir.frame_pointer<all>

// CIR-RESERVED: module {{.*}}attributes {{.*}}cir.frame_pointer = #cir.frame_pointer<reserved>
// CIR-RESERVED: cir.func {{.*}}@_Z10definitionv() attributes {{.*}}frame_pointer = #cir.frame_pointer<reserved>
// CIR-RESERVED: cir.func private @_Z11declarationv() attributes {{.*}}frame_pointer = #cir.frame_pointer<reserved>

// CIR-NLNR: module {{.*}}attributes {{.*}}cir.frame_pointer = #cir.frame_pointer<non_leaf_no_reserve>
// CIR-NLNR: cir.func {{.*}}@_Z10definitionv() attributes {{.*}}frame_pointer = #cir.frame_pointer<non_leaf_no_reserve>
// CIR-NLNR: cir.func private @_Z11declarationv() attributes {{.*}}frame_pointer = #cir.frame_pointer<non_leaf_no_reserve>

// LLVM-NONE: define {{.*}}void @_Z10definitionv()
// LLVM-NONE-NOT: "frame-pointer"

// LLVM-NON-LEAF: define {{.*}}void @_Z10definitionv() #[[DEF_ATTR:[0-9]+]]
// LLVM-NON-LEAF: declare void @_Z11declarationv() #[[DECL_ATTR:[0-9]+]]
// LLVM-NON-LEAF: attributes #[[DEF_ATTR]] = {{.*}}"frame-pointer"="non-leaf"
// LLVM-NON-LEAF: attributes #[[DECL_ATTR]] = {{.*}}"frame-pointer"="non-leaf"
// LLVM-NON-LEAF: !{i32 7, !"frame-pointer", i32 1}

// LLVM-ALL: define {{.*}}void @_Z10definitionv() #[[DEF_ATTR:[0-9]+]]
// LLVM-ALL: call void @_Z11declarationv(){{$}}
// LLVM-ALL: declare void @_Z11declarationv() #[[DECL_ATTR:[0-9]+]]
// LLVM-ALL: attributes #[[DEF_ATTR]] = {{.*}}"frame-pointer"="all"
// LLVM-ALL: attributes #[[DECL_ATTR]] = {{.*}}"frame-pointer"="all"
// LLVM-ALL: !{i32 7, !"frame-pointer", i32 2}

// LLVM-RESERVED: define {{.*}}void @_Z10definitionv() #[[DEF_ATTR:[0-9]+]]
// LLVM-RESERVED: declare void @_Z11declarationv() #[[DECL_ATTR:[0-9]+]]
// LLVM-RESERVED: attributes #[[DEF_ATTR]] = {{.*}}"frame-pointer"="reserved"
// LLVM-RESERVED: attributes #[[DECL_ATTR]] = {{.*}}"frame-pointer"="reserved"
// LLVM-RESERVED: !{i32 7, !"frame-pointer", i32 3}

// LLVM-NLNR: define {{.*}}void @_Z10definitionv() #[[DEF_ATTR:[0-9]+]]
// LLVM-NLNR: declare void @_Z11declarationv() #[[DECL_ATTR:[0-9]+]]
// LLVM-NLNR: attributes #[[DEF_ATTR]] = {{.*}}"frame-pointer"="non-leaf-no-reserve"
// LLVM-NLNR: attributes #[[DECL_ATTR]] = {{.*}}"frame-pointer"="non-leaf-no-reserve"
// LLVM-NLNR: !{i32 7, !"frame-pointer", i32 4}
