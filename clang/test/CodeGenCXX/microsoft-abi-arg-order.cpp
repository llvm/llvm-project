// RUN: %clang_cc1 -mconstructor-aliases -std=c++11 -fexceptions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s -check-prefix=X86
// RUN: %clang_cc1 -mconstructor-aliases -std=c++11 -fexceptions -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s -check-prefix=X64

struct A {
  A(int a);
  A(const A &o);
  ~A();
  int a;
};

void foo(A a, A b, A c) {
}

// Order of destruction should be left to right.
//
// X86-LABEL: define dso_local void @"?foo@@YAXUA@@00@Z"
// X86:          (ptr inalloca([[argmem_ty:<{ %struct.A, %struct.A, %struct.A }>]]) %0)
// X86: %[[a:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %0, i32 0, i32 0
// X86: %[[b:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %0, i32 0, i32 1
// X86: %[[c:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %0, i32 0, i32 2
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[a]])
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[b]])
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[c]])
// X86: ret void

// X64-LABEL: define dso_local void @"?foo@@YAXUA@@00@Z"
// X64:         (ptr noundef %[[a:[^,]*]], ptr noundef %[[b:[^,]*]], ptr noundef %[[c:[^)]*]])
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[a]])
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[b]])
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[c]])
// X64: ret void


void call_foo() {
  foo(A(1), A(2), A(3));
}

// Order of evaluation should be right to left, and we should clean up the right
// things as we unwind.
//
// X86-LABEL: define dso_local void @"?call_foo@@YAXXZ"()
// X86: call ptr @llvm.stacksave()
// X86: %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty]]
// X86: %[[arg3:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 2
// X86: call x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg3]], i32 noundef 3)
// X86: %[[arg2:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 1
// X86: invoke x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg2]], i32 noundef 2)
// X86: %[[arg1:[^ ]*]] = getelementptr inbounds [[argmem_ty]], ptr %[[argmem]], i32 0, i32 0
// X86: invoke x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg1]], i32 noundef 1)
// X86: call void @"?foo@@YAXUA@@00@Z"(ptr inalloca([[argmem_ty]]) %[[argmem]])
// X86: call void @llvm.stackrestore
// X86: ret void
//
//   lpad2:
// X86: cleanuppad within none []
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[arg2]])
// X86: cleanupret
//
//   ehcleanup:
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[arg3]])

// X64-LABEL: define dso_local void @"?call_foo@@YAXXZ"()
// X64: call noundef ptr @"??0A@@QEAA@H@Z"(ptr {{[^,]*}} %[[arg3:[^,]*]], i32 noundef 3)
// X64: invoke noundef ptr @"??0A@@QEAA@H@Z"(ptr {{[^,]*}} %[[arg2:[^,]*]], i32 noundef 2)
// X64: invoke noundef ptr @"??0A@@QEAA@H@Z"(ptr {{[^,]*}} %[[arg1:[^,]*]], i32 noundef 1)
// X64: call void @"?foo@@YAXUA@@00@Z"
// X64:       (ptr noundef %[[arg1]], ptr noundef %[[arg2]], ptr noundef %[[arg3]])
// X64: ret void
//
//   lpad2:
// X64: cleanuppad within none []
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[arg2]])
// X64: cleanupret
//
//   ehcleanup:
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[arg3]])
