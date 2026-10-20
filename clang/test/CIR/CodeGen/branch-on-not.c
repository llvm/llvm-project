// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

void f(void);
void h(void);
int g(int *p, int y);
int t(void);
int u(void);

void if_not(int *p) {
  if (!p)
    f();
}

// CIR-LABEL: cir.func{{.*}} @if_not(
// CIR:         %[[B:.*]] = cir.cast ptr_to_bool %{{.*}} : !cir.ptr<!s32i> -> !cir.bool
// CIR-NEXT:    %[[N:.*]] = cir.not %[[B]] : !cir.bool
// CIR-NEXT:    cir.if %[[N]] {

// LLVM-LABEL: define dso_local void @if_not(
// LLVM:         %[[NONNULL:.*]] = icmp ne ptr %{{.*}}, null
// LLVM-NEXT:    br i1 %[[NONNULL]], label %[[END:[a-z0-9.]+]], label %[[THEN:[a-z0-9.]+]]
// LLVM:       {{^}}[[THEN]]:
// LLVM-NEXT:    call void @f()
// LLVM-NEXT:    br label %[[END]]

void if_not_else(int *p) {
  if (!p)
    f();
  else
    h();
}

// CIR-LABEL: cir.func{{.*}} @if_not_else(
// CIR:         %[[N:.*]] = cir.not %{{.*}} : !cir.bool
// CIR-NEXT:    cir.if %[[N]] {

// LLVM-LABEL: define dso_local void @if_not_else(
// LLVM:         %[[NONNULL:.*]] = icmp ne ptr %{{.*}}, null
// LLVM-NEXT:    br i1 %[[NONNULL]], label %[[ELSE:[a-z0-9.]+]], label %[[THEN:[a-z0-9.]+]]
// LLVM:       {{^}}[[THEN]]:
// LLVM-NEXT:    call void @f()
// LLVM:       {{^}}[[ELSE]]:
// LLVM-NEXT:    call void @h()

int not_ternary(int a) { return !a ? t() : u(); }

// CIR-LABEL: cir.func{{.*}} @not_ternary(
// CIR:         %[[N:.*]] = cir.not %{{.*}} : !cir.bool
// CIR-NEXT:    %{{.*}} = cir.ternary(%[[N]], true {

// LLVM-LABEL: define dso_local i32 @not_ternary(
// LLVM:         %[[NONZERO:.*]] = icmp ne i32 %{{.*}}, 0
// LLVM-NEXT:    br i1 %[[NONZERO]], label %[[FALSE:[a-z0-9.]+]], label %[[TRUE:[a-z0-9.]+]]
// LLVM:       {{^}}[[TRUE]]:
// LLVM-NEXT:    %{{.*}} = call i32 @t()
// LLVM:       {{^}}[[FALSE]]:
// LLVM-NEXT:    %{{.*}} = call i32 @u()

void while_not(volatile int *done) {
  while (!*done)
    f();
}

// CIR-LABEL: cir.func{{.*}} @while_not(
// CIR:         %[[B:.*]] = cir.cast int_to_bool %{{.*}} : !s32i -> !cir.bool
// CIR-NEXT:    %[[N:.*]] = cir.not %[[B]] : !cir.bool
// CIR-NEXT:    cir.condition(%[[N]])

// LLVM-LABEL: define dso_local void @while_not(
// LLVM:         %[[NONZERO:.*]] = icmp ne i32 %{{.*}}, 0
// LLVM-NEXT:    %[[LNOT:.*]] = xor i1 %[[NONZERO]], true
// LLVM-NEXT:    br i1 %[[LNOT]], label %[[BODY:[a-z0-9.]+]], label %{{[a-z0-9.]+}}
// LLVM:       {{^}}[[BODY]]:
// LLVM-NEXT:    call void @f()

int not_or(int *p, int y) {
  if (!p || *p == y)
    return g(p, y);
  return 0;
}

// CIR-LABEL: cir.func{{.*}} @not_or(
// CIR:         %[[B:.*]] = cir.cast ptr_to_bool %{{.*}} : !cir.ptr<!s32i> -> !cir.bool
// CIR-NEXT:    %[[N:.*]] = cir.not %[[B]] : !cir.bool
// CIR-NEXT:    %{{.*}} = cir.ternary(%[[N]], true {

// LLVM-LABEL: define dso_local i32 @not_or(
// LLVM:         %[[NONNULL:.*]] = icmp ne ptr %{{.*}}, null
// LLVM-NEXT:    br i1 %[[NONNULL]], label %[[RHS:[a-z0-9.]+]], label %{{[a-z0-9.]+}}
// LLVM:       {{^}}[[RHS]]:
// LLVM:         %[[CMP:.*]] = icmp eq i32
// LLVMCIR:      %[[OR:.*]] = phi i1 [ true, %{{.*}} ], [ %[[CMP]], %[[RHS]] ]
// LLVMCIR:      br i1 %[[OR]], label %[[THEN:[a-z0-9.]+]], label %{{[a-z0-9.]+}}
// OGCG-NEXT:    br i1 %[[CMP]], label %[[THEN:[a-z0-9.]+]], label %{{[a-z0-9.]+}}
// LLVM:       {{^}}[[THEN]]:
// LLVM:         call i32 @g(ptr noundef %{{.+}}, i32 noundef %{{.+}})

void not_of_or(int a, int b) {
  if (!(a || b))
    f();
}

// CIR-LABEL: cir.func{{.*}} @not_of_or(
// CIR:         %[[OR:.*]] = cir.ternary(
// CIR:         %[[N:.*]] = cir.not %[[OR]] : !cir.bool
// CIR-NEXT:    cir.if %[[N]] {

// LLVM-LABEL: define dso_local void @not_of_or(
// LLVMCIR:      %[[OR:.*]] = phi i1
// LLVMCIR-NOT:  xor
// LLVMCIR:      br i1 %[[OR]], label %{{[a-z0-9.]+}}, label %[[THEN:[a-z0-9.]+]]
// OGCG:         br i1 %{{.*}}, label %[[END:[a-z0-9.]+]], label %{{[a-z0-9.]+}}
// OGCG:         br i1 %{{.*}}, label %[[END]], label %[[THEN:[a-z0-9.]+]]
// LLVM:       {{^}}[[THEN]]:
// LLVM-NEXT:    call void @f()
