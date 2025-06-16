// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  int x;
  int y;
};

void f1(struct S);
void f2() {
  struct S s;
  f1(s);
}

// CIR-LABEL: cir.func @f2()
// CIR:         %[[S:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!rec_S>, !rec_S
// CIR-NEXT:    cir.call @f1(%[[S]]) : (!rec_S) -> ()

// LLVM-LABEL: define void @f2()
// LLVM:         %[[S:.+]] = load %struct.S, ptr %{{.+}}, align 4
// LLVM-NEXT:    call void @f1(%struct.S %[[S]])

// OGCG-LABEL: define dso_local void @f2()
// OGCG:         %[[S:.+]] = load i64, ptr %{{.+}}, align 4
// OGCG-NEXT:    call void @f1(i64 %[[S]])

struct S f3();
void f4() {
  struct S s = f3();
}

// CIR-LABEL: cir.func @f4() {
// CIR:         %[[S:.+]] = cir.call @f3() : () -> !rec_S
// CIR-NEXT:    cir.store align(4) %[[S]], %{{.+}} : !rec_S, !cir.ptr<!rec_S>

// LLVM-LABEL: define void @f4() {
// LLVM:         %[[S:.+]] = call %struct.S (...) @f3()
// LLVM-NEXT:    store %struct.S %[[S]], ptr %{{.+}}, align 4

// OGCG-LABEL: define dso_local void @f4() #0 {
// OGCG:         %[[S:.+]] = call i64 (...) @f3()
// OGCG-NEXT:    store i64 %[[S]], ptr %{{.+}}, align 4

struct Big {
  int data[10];
};

void f5(struct Big);
struct Big f6();

void f7() {
  struct Big b;
  f5(b);
}

// CIR-LABEL: cir.func @f7()
// CIR:         %[[B:.+]] = cir.load align(4) %{{.+}} : !cir.ptr<!rec_Big>, !rec_Big
// CIR-NEXT:    cir.call @f5(%[[B]]) : (!rec_Big) -> ()

// LLVM-LABEL: define void @f7() {
// LLVM:         %[[B:.+]] = load %struct.Big, ptr %{{.+}}, align 4
// LLVM-NEXT:    call void @f5(%struct.Big %[[B]])

// OGCG-LABEL: define dso_local void @f7() #0 {
// OGCG:         %[[B:.+]] = alloca %struct.Big, align 8
// OGCG-NEXT:    call void @f5(ptr noundef byval(%struct.Big) align 8 %[[B]])

void f8() {
  struct Big b = f6();
}

// CIR-LABEL: cir.func @f8()
// CIR:         %[[B:.+]] = cir.call @f6() : () -> !rec_Big
// CIR:         cir.store align(4) %[[B]], %{{.+}} : !rec_Big, !cir.ptr<!rec_Big>

// LLVM-LABEL: define void @f8() {
// LLVM:        %[[B:.+]] = call %struct.Big (...) @f6()
// LLVM-NEXT:   store %struct.Big %[[B]], ptr %{{.+}}, align 4

// OGCG-LABEL: define dso_local void @f8() #0 {
// OGCG:         %[[B:.+]] = alloca %struct.Big, align 4
// OGCG-NEXT:    call void (ptr, ...) @f6(ptr dead_on_unwind writable sret(%struct.Big) align 4 %[[B]])

void f9() {
  f1(f3());
}

// CIR-LABEL: cir.func @f9()
// CIR:         %[[SLOT:.+]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["agg.tmp0"] {alignment = 4 : i64}
// CIR-NEXT:    %[[RET:.+]] = cir.call @f3() : () -> !rec_S
// CIR-NEXT:    cir.store align(4) %[[RET]], %[[SLOT]] : !rec_S, !cir.ptr<!rec_S>
// CIR-NEXT:    %[[ARG:.+]] = cir.load align(4) %[[SLOT]] : !cir.ptr<!rec_S>, !rec_S
// CIR-NEXT:    cir.call @f1(%[[ARG]]) : (!rec_S) -> ()

// LLVM-LABEL: define void @f9() {
// LLVM:         %[[SLOT:.+]] = alloca %struct.S, i64 1, align 4
// LLVM-NEXT:    %[[RET:.+]] = call %struct.S (...) @f3()
// LLVM-NEXT:    store %struct.S %[[RET]], ptr %[[SLOT]], align 4
// LLVM-NEXT:    %[[ARG:.+]] = load %struct.S, ptr %[[SLOT]], align 4
// LLVM-NEXT:    call void @f1(%struct.S %[[ARG]])

// OGCG-LABEL: define dso_local void @f9() #0 {
// OGCG:         %[[SLOT:.+]] = alloca %struct.S, align 4
// OGCG-NEXT:    %[[RET:.+]] = call i64 (...) @f3()
// OGCG-NEXT:    store i64 %[[RET]], ptr %[[SLOT]], align 4
// OGCG-NEXT:    %[[ARG:.+]] = load i64, ptr %[[SLOT]], align 4
// OGCG-NEXT:    call void @f1(i64 %[[ARG]])
