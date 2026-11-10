// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-NOEH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s -check-prefixes=CIR,CIR-EH
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// The elements of an array that an initialization has constructed so far are
// destroyed when the initialization is left early, here by a return in a
// statement expression, and not only when an initializer throws. With or
// without exceptions, the partial array destroy is a normal cleanup guarded by
// a flag that the end of the initialization clears.

struct Obj {
  Obj(char);
  ~Obj();
};

void early_return(bool c) {
  Obj arr[2] = {Obj{'a'}, ({
                  if (c)
                    return;
                  Obj{'b'};
                })};
}

// CIR-LABEL: cir.func {{.*}} @_Z12early_returnb(
// CIR:         %[[ARR:.*]] = cir.alloca "arr"
// CIR:         %[[END:.*]] = cir.alloca "arrayinit.endOfInit"
// CIR:         %[[ACTIVE:.*]] = cir.alloca "cleanup.isactive"
// CIR:         %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR]]
// CIR-NEXT:    cir.store {{.*}} %[[BEGIN]], %[[END]]
// CIR-NEXT:    %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:    cir.store %[[TRUE]], %[[ACTIVE]]
// CIR-NEXT:    cir.cleanup.scope {
// CIR:           cir.call @_ZN3ObjC1Ec(%[[BEGIN]],
// CIR:           %[[SECOND:.*]] = cir.ptr_stride %[[BEGIN]]
// CIR-NEXT:      cir.store {{.*}} %[[SECOND]], %[[END]]
// CIR:           cir.if
// CIR-NEXT:        cir.return
// CIR:           cir.call @_ZN3ObjC1Ec(%[[SECOND]],
// CIR:           %[[FALSE:.*]] = cir.const #false
// CIR-NEXT:      cir.store %[[FALSE]], %[[ACTIVE]]
// CIR-NEXT:      cir.yield
// CIR-NOEH-NEXT: } cleanup normal {
// CIR-EH-NEXT:   } cleanup all {
// CIR-NEXT:      %[[IS_ACTIVE:.*]] = cir.load {{.*}} %[[ACTIVE]]
// CIR-NEXT:      cir.if %[[IS_ACTIVE]] {
// CIR-NEXT:        %[[CUR_END:.*]] = cir.load {{.*}} %[[END]]
// CIR-NEXT:        %[[NOT_EMPTY:.*]] = cir.cmp ne %[[CUR_END]], %[[BEGIN]]
// CIR-NEXT:        cir.if %[[NOT_EMPTY]] {
// CIR:               cir.do {
// CIR:                 cir.call @_ZN3ObjD1Ev(
// CIR:         cir.return

// The return leaves through the cleanup, which destroys the first element.
//
// LLVM-LABEL: define dso_local void @_Z12early_returnb(
// LLVM:         store i8 1, ptr %[[ACTIVE:[0-9]+]]
// LLVM:         call void @_ZN3ObjC1Ec(ptr {{.*}}, i8 noundef signext 97)
// LLVM:         br i1 %{{.*}}, label %[[RETURN:[0-9]+]], label %{{.*}}
// LLVM:       [[RETURN]]:
// LLVM-NEXT:    store i32 1, ptr %[[SLOT:[0-9]+]]
// LLVM-NEXT:    br label %[[CLEANUP:[0-9]+]]
// LLVM:         call void @_ZN3ObjC1Ec(ptr {{.*}}, i8 noundef signext 98)
// LLVM:         store i8 0, ptr %[[ACTIVE]]
// LLVM-NEXT:    store i32 0, ptr %[[SLOT]]
// LLVM-NEXT:    br label %[[CLEANUP]]
// LLVM:       [[CLEANUP]]:
// LLVM-NEXT:    %[[FLAG:.*]] = load i8, ptr %[[ACTIVE]]
// LLVM-NEXT:    %[[IS_ACTIVE:.*]] = trunc i8 %[[FLAG]] to i1
// LLVM-NEXT:    br i1 %[[IS_ACTIVE]], label %{{.*}}, label %{{.*}}
// LLVM:         call void @_ZN3ObjD1Ev(
// LLVM:         ret void

// OGCG-LABEL: define dso_local void @_Z12early_returnb(
// OGCG:       if.then:
// OGCG-NEXT:    load ptr, ptr %arrayinit.endOfInit
// OGCG:       arraydestroy.body:
// OGCG:         call void @_ZN3ObjD1Ev(

// Without an early exit, nothing reaches the cleanup on the normal path. It
// stays an EH cleanup with exceptions, and without them it is dropped together
// with the end of the initialized part that it would read.

void no_exit() { Obj arr[2] = {Obj{'a'}, Obj{'b'}}; }

// CIR-LABEL: cir.func {{.*}} @_Z7no_exitv(
// CIR-NOEH-NOT: arrayinit.endOfInit
// CIR-NOEH-NOT: cleanup.isactive
// CIR-EH:        %[[END:.*]] = cir.alloca "arrayinit.endOfInit"
// CIR-EH:        cir.cleanup.scope {
// CIR-EH:          cir.call @_ZN3ObjC1Ec(
// CIR-EH:          cir.call @_ZN3ObjC1Ec(
// CIR-EH:        } cleanup eh {
// CIR-EH-NEXT:     cir.load {{.*}} %[[END]]
// CIR:           cir.call @_ZN3ObjD1Ev(
// CIR:         cir.return

// LLVM-LABEL: define dso_local void @_Z7no_exitv(
// LLVM-NOT:     store i8 1
// LLVM:         ret void
