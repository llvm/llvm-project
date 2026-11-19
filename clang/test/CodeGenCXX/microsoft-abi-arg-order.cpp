// RUN: %clang_cc1 -mconstructor-aliases -std=c++11 -fexceptions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s -check-prefix=X86
// RUN: %clang_cc1 -mconstructor-aliases -std=c++11 -fexceptions -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s -check-prefix=X64
// RUN: %clang_cc1 -mconstructor-aliases -std=c++23 -fexceptions -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s -check-prefix=X86-CXX23
// RUN: %clang_cc1 -mconstructor-aliases -std=c++23 -fexceptions -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s -check-prefix=X64-CXX23

struct A {
  A(int a);
  A(const A &o);
  ~A();
  int a;
};

void foo(A a, A b, A c) {
}

// Parameter destruction is left to right.
//
// X86-LABEL: define dso_local void @"?foo@@YAXUA@@00@Z"
// X86:          (ptr inalloca([[argmem_ty:<{ %struct.A, %struct.A, %struct.A }>]]) %0)
// X86: %[[a:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %0, i32 0, i32 0
// X86: %[[b:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %0, i32 0, i32 1
// X86: %[[c:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %0, i32 0, i32 2
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[a]])
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[b]])
// X86: call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[c]])
// X86: ret void

// X64-LABEL: define dso_local void @"?foo@@YAXUA@@00@Z"
// X64:         (ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[a:[^,]*]], ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[b:[^,]*]], ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[c:[^)]*]])
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
// X86: call ptr @llvm.stacksave.p0()
// X86: %[[argmem:[^ ]*]] = alloca inalloca [[argmem_ty]]
// X86: %[[arg3:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %[[argmem]], i32 0, i32 2
// X86: call x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg3]], i32 noundef 3)
// X86: %[[arg2:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %[[argmem]], i32 0, i32 1
// X86: invoke x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg2]], i32 noundef 2)
// X86: %[[arg1:[^ ]*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %[[argmem]], i32 0, i32 0
// X86: invoke x86_thiscallcc noundef ptr @"??0A@@QAE@H@Z"(ptr {{[^,]*}} %[[arg1]], i32 noundef 1)
// X86: call void @"?foo@@YAXUA@@00@Z"(ptr inalloca([[argmem_ty]]) %[[argmem]])
// X86: call void @llvm.stackrestore.p0
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
// X64:       (ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[arg1]], ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[arg2]], ptr nofreeobj noundef align 4 dead_on_return dereferenceable(4) %[[arg3]])
// X64: ret void
//
//   lpad2:
// X64: cleanuppad within none []
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[arg2]])
// X64: cleanupret
//
//   ehcleanup:
// X64: call void @"??1A@@QEAA@XZ"(ptr {{[^,]*}} %[[arg3]])

#if __cplusplus >= 202302L
struct B {
  B(int b);
  B(const B &o);
  ~B();
  int b;
  void operator[](this B self, B i, B j);
  void operator()(this B self, B i, B j);
};

void B::operator[](this B self, B i, B j) {
}

// Parameter destruction is left to right.
//
// X86-CXX23-LABEL: define dso_local void @"??AB@@SAX_VU0@00@Z"
// X86-CXX23:          (ptr inalloca([[argmem_b:<{ %struct.B, %struct.B, %struct.B }>]]) %0)
// X86-CXX23: %[[self:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %0, i32 0, i32 0
// X86-CXX23: %[[i:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %0, i32 0, i32 1
// X86-CXX23: %[[j:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %0, i32 0, i32 2
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[self]])
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[i]])
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[j]])
// X86-CXX23: ret void

// X64-CXX23-LABEL: define dso_local void @"??AB@@SAX_VU0@00@Z"
// X64-CXX23:         (ptr {{[^,]*}} %[[self:[^,]*]], ptr {{[^,]*}} %[[i:[^,]*]], ptr {{[^,]*}} %[[j:[^)]*]])
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[self]])
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[i]])
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[j]])
// X64-CXX23: ret void


void call_subscript() {
  B(1)[B(2), B(3)];
}

// X86-CXX23-LABEL: define dso_local void @"?call_subscript@@YAXXZ"()
// X86-CXX23: %[[argmem:[^ ]*]] = alloca inalloca [[argmem_b]]
// X86-CXX23: %[[obj:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 0
// X86-CXX23: call x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[obj]], i32 noundef 1)
// X86-CXX23: %[[idx1:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 1
// X86-CXX23: invoke x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[idx1]], i32 noundef 2)
// X86-CXX23: to label %{{[^ ]+}} unwind label %[[OBJ_CLEANUP:[^ ]+]]
// X86-CXX23: %[[idx2:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 2
// X86-CXX23: invoke x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[idx2]], i32 noundef 3)
// X86-CXX23: to label %{{[^ ]+}} unwind label %[[IDX_CLEANUP:[^ ]+]]
// X86-CXX23: call void @"??AB@@SAX_VU0@00@Z"(ptr inalloca([[argmem_b]]) %[[argmem]])
// X86-CXX23: ret void
//
// X86-CXX23: [[IDX_CLEANUP]]:
// X86-CXX23: cleanuppad within none []
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[idx1]])
// X86-CXX23: cleanupret {{.*}} unwind label %[[OBJ_CLEANUP]]
//
// X86-CXX23: [[OBJ_CLEANUP]]:
// X86-CXX23: cleanuppad within none []
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[obj]])

// X64-CXX23-LABEL: define dso_local void @"?call_subscript@@YAXXZ"()
// X64-CXX23: %[[obj:[^ ]*]] = alloca %struct.B
// X64-CXX23: %[[idx1:[^ ]*]] = alloca %struct.B
// X64-CXX23: %[[idx2:[^ ]*]] = alloca %struct.B
// X64-CXX23: call noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[obj]], i32 noundef 1)
// X64-CXX23: invoke noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[idx1]], i32 noundef 2)
// X64-CXX23: to label %{{[^ ]+}} unwind label %[[OBJ_CLEANUP:[^ ]+]]
// X64-CXX23: invoke noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[idx2]], i32 noundef 3)
// X64-CXX23: to label %{{[^ ]+}} unwind label %[[IDX_CLEANUP:[^ ]+]]
// X64-CXX23: call void @"??AB@@SAX_VU0@00@Z"
// X64-CXX23:       (ptr {{[^,]*}} %[[obj]], ptr {{[^,]*}} %[[idx1]], ptr {{[^,]*}} %[[idx2]])
// X64-CXX23: ret void
//
// X64-CXX23: [[IDX_CLEANUP]]:
// X64-CXX23: cleanuppad within none []
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[idx1]])
// X64-CXX23: cleanupret {{.*}} unwind label %[[OBJ_CLEANUP]]
//
// X64-CXX23: [[OBJ_CLEANUP]]:
// X64-CXX23: cleanuppad within none []
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[obj]])

void call_operator() {
  B(1)(B(2), B(3));
}

// The call arguments must occupy their declared parameter slots.
//
// X86-CXX23-LABEL: define dso_local void @"?call_operator@@YAXXZ"()
// X86-CXX23: %[[argmem:[^ ]*]] = alloca inalloca [[argmem_b]]
// X86-CXX23: %[[obj:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 0
// X86-CXX23: call x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[obj]], i32 noundef 1)
// X86-CXX23: %[[arg2:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 2
// X86-CXX23: invoke x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[arg2]], i32 noundef 3)
// X86-CXX23: to label %{{[^ ]+}} unwind label %[[OBJ_CLEANUP:[^ ]+]]
// X86-CXX23: %[[arg1:[^ ]*]] = getelementptr inbounds nuw [[argmem_b]], ptr %[[argmem]], i32 0, i32 1
// X86-CXX23: invoke x86_thiscallcc noundef ptr @"??0B@@QAE@H@Z"(ptr {{[^,]*}} %[[arg1]], i32 noundef 2)
// X86-CXX23: to label %{{[^ ]+}} unwind label %[[ARG_CLEANUP:[^ ]+]]
// X86-CXX23: call void @"??RB@@SAX_VU0@00@Z"(ptr inalloca([[argmem_b]]) %[[argmem]])
// X86-CXX23: ret void
//
// X86-CXX23: [[ARG_CLEANUP]]:
// X86-CXX23: cleanuppad within none []
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[arg2]])
// X86-CXX23: cleanupret {{.*}} unwind label %[[OBJ_CLEANUP]]
//
// X86-CXX23: [[OBJ_CLEANUP]]:
// X86-CXX23: cleanuppad within none []
// X86-CXX23: call x86_thiscallcc void @"??1B@@QAE@XZ"(ptr {{[^,]*}} %[[obj]])

// X64-CXX23-LABEL: define dso_local void @"?call_operator@@YAXXZ"()
// X64-CXX23: %[[obj:[^ ]*]] = alloca %struct.B
// X64-CXX23: %[[arg2:[^ ]*]] = alloca %struct.B
// X64-CXX23: %[[arg1:[^ ]*]] = alloca %struct.B
// X64-CXX23: call noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[obj]], i32 noundef 1)
// X64-CXX23: invoke noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[arg2]], i32 noundef 3)
// X64-CXX23: to label %{{[^ ]+}} unwind label %[[OBJ_CLEANUP:[^ ]+]]
// X64-CXX23: invoke noundef ptr @"??0B@@QEAA@H@Z"(ptr {{[^,]*}} %[[arg1]], i32 noundef 2)
// X64-CXX23: to label %{{[^ ]+}} unwind label %[[ARG_CLEANUP:[^ ]+]]
// X64-CXX23: call void @"??RB@@SAX_VU0@00@Z"
// X64-CXX23:       (ptr {{[^,]*}} %[[obj]], ptr {{[^,]*}} %[[arg1]], ptr {{[^,]*}} %[[arg2]])
// X64-CXX23: ret void
//
// X64-CXX23: [[ARG_CLEANUP]]:
// X64-CXX23: cleanuppad within none []
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[arg2]])
// X64-CXX23: cleanupret {{.*}} unwind label %[[OBJ_CLEANUP]]
//
// X64-CXX23: [[OBJ_CLEANUP]]:
// X64-CXX23: cleanuppad within none []
// X64-CXX23: call void @"??1B@@QEAA@XZ"(ptr {{[^,]*}} %[[obj]])

struct SizeObject {
  operator SizeObject *();
  int operator[](this SizeObject *const Self
                     __attribute__((pass_object_size(0))),
                 int Index);
  int operator()(this SizeObject *const Self
                    __attribute__((pass_object_size(0))),
                 SizeObject *const Arg __attribute__((pass_object_size(0))),
                 int Index);
};
SizeObject make_size_object();
SizeObject *make_size_arg();
int make_size_index();

int pass_object_size_for_object() {
  return make_size_object()[make_size_index()];
}

// X86-CXX23-LABEL: define dso_local noundef i32 @"?pass_object_size_for_object@@YAHXZ"()
// X86-CXX23: call void @"?make_size_object@@YA?AUSizeObject@@XZ"()
// X86-CXX23: %[[SELF:[^ ]+]] = call x86_thiscallcc noundef ptr @"??BSizeObject@@QAEPAU0@XZ"
// X86-CXX23: %[[SIZE:[^ ]+]] = call i32 @llvm.objectsize.i32.p0(ptr %[[SELF]],
// X86-CXX23: %[[INDEX:[^ ]+]] = call noundef i32 @"?make_size_index@@YAHXZ"()
// X86-CXX23: call noundef i32 @"??ASizeObject@@SAH_VQAU0@W4__pass_object_size0@__clang@@H@Z"(ptr noundef %[[SELF]], i32 noundef %[[SIZE]], i32 noundef %[[INDEX]])

// X64-CXX23-LABEL: define dso_local noundef i32 @"?pass_object_size_for_object@@YAHXZ"()
// X64-CXX23: call i8 @"?make_size_object@@YA?AUSizeObject@@XZ"()
// X64-CXX23: %[[SELF:[^ ]+]] = call noundef ptr @"??BSizeObject@@QEAAPEAU0@XZ"
// X64-CXX23: %[[SIZE:[^ ]+]] = call i64 @llvm.objectsize.i64.p0(ptr %[[SELF]],
// X64-CXX23: %[[INDEX:[^ ]+]] = call noundef i32 @"?make_size_index@@YAHXZ"()
// X64-CXX23: call noundef i32 @"??ASizeObject@@SAH_VQEAU0@W4__pass_object_size0@__clang@@H@Z"(ptr noundef %[[SELF]], i64 noundef %[[SIZE]], i32 noundef %[[INDEX]])

int pass_object_size_for_call() {
  return make_size_object()(make_size_arg(), make_size_index());
}

// X86-CXX23-LABEL: define dso_local noundef i32 @"?pass_object_size_for_call@@YAHXZ"()
// X86-CXX23: call void @"?make_size_object@@YA?AUSizeObject@@XZ"()
// X86-CXX23: %[[SELF:[^ ]+]] = call x86_thiscallcc noundef ptr @"??BSizeObject@@QAEPAU0@XZ"
// X86-CXX23: %[[SELF_SIZE:[^ ]+]] = call i32 @llvm.objectsize.i32.p0(ptr %[[SELF]],
// X86-CXX23: %[[INDEX:[^ ]+]] = call noundef i32 @"?make_size_index@@YAHXZ"()
// X86-CXX23: %[[ARG:[^ ]+]] = call noundef ptr @"?make_size_arg@@YAPAUSizeObject@@XZ"()
// X86-CXX23: %[[ARG_SIZE:[^ ]+]] = call i32 @llvm.objectsize.i32.p0(ptr %[[ARG]],
// X86-CXX23: call noundef i32 @"??RSizeObject@@SAH_VQAU0@W4__pass_object_size0@__clang@@01H@Z"(ptr noundef %[[SELF]], i32 noundef %[[SELF_SIZE]], ptr noundef %[[ARG]], i32 noundef %[[ARG_SIZE]], i32 noundef %[[INDEX]])

// X64-CXX23-LABEL: define dso_local noundef i32 @"?pass_object_size_for_call@@YAHXZ"()
// X64-CXX23: call i8 @"?make_size_object@@YA?AUSizeObject@@XZ"()
// X64-CXX23: %[[SELF:[^ ]+]] = call noundef ptr @"??BSizeObject@@QEAAPEAU0@XZ"
// X64-CXX23: %[[SELF_SIZE:[^ ]+]] = call i64 @llvm.objectsize.i64.p0(ptr %[[SELF]],
// X64-CXX23: %[[INDEX:[^ ]+]] = call noundef i32 @"?make_size_index@@YAHXZ"()
// X64-CXX23: %[[ARG:[^ ]+]] = call noundef ptr @"?make_size_arg@@YAPEAUSizeObject@@XZ"()
// X64-CXX23: %[[ARG_SIZE:[^ ]+]] = call i64 @llvm.objectsize.i64.p0(ptr %[[ARG]],
// X64-CXX23: call noundef i32 @"??RSizeObject@@SAH_VQEAU0@W4__pass_object_size0@__clang@@01H@Z"(ptr noundef %[[SELF]], i64 noundef %[[SELF_SIZE]], ptr noundef %[[ARG]], i64 noundef %[[ARG_SIZE]], i32 noundef %[[INDEX]])
#endif
