// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
struct S {
    int x;    
};

// LLVM: define void @zeroInit
// LLVM: [[TMP0:%.*]] = alloca %struct.S, i64 1
// LLVM: store %struct.S zeroinitializer, ptr [[TMP0]]
void zeroInit() {
  struct S s = {0};
}
