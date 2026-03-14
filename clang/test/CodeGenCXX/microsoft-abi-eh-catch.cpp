// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:     -mconstructor-aliases -fexceptions -fcxx-exceptions \
// RUN:     -O1 -disable-llvm-passes \
// RUN:     | FileCheck -check-prefix WIN64 %s

extern "C" void might_throw();

// Simplify the generated IR with noexcept.
extern "C" void recover() noexcept(true);
extern "C" void handle_exception(void *e) noexcept(true);

extern "C" void catch_all() {
  try {
    might_throw();
  } catch (...) {
    recover();
  }
}

// WIN64-LABEL: define dso_local void @catch_all()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont:[^ ]*]] unwind label %[[catchswitch_lpad:[^ ]*]]
//
// WIN64: [[catchswitch_lpad]]
// WIN64: %[[catchswitch:[^ ]*]] = catchswitch within none [label %[[catchpad_lpad:[^ ]*]]] unwind to caller
//
// WIN64: [[catchpad_lpad]]
// WIN64: catchpad within %[[catchswitch]] [ptr null, i32 64, ptr null]
// WIN64: call void @recover()
// WIN64: catchret from %{{.*}} to label %[[catchret:[^ ]*]]
//
// WIN64: [[catchret]]
// WIN64-NEXT: br label %[[ret:[^ ]*]]
//
// WIN64: [[ret]]
// WIN64: ret void
//
// WIN64: [[cont]]
// WIN64: br label %[[ret]]

extern "C" void catch_int() {
  try {
    might_throw();
  } catch (int e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define dso_local void @catch_int()
// WIN64: catchpad within %{{[^ ]*}} [ptr @"??_R0H@8", i32 0, ptr %[[e_addr:.*]]]
//
// The catchpad instruction starts the lifetime of 'e'. Unfortunately, that
// leaves us with nowhere to put lifetime.start, so we don't emit lifetime
// markers for now.
// WIN64-NOT: lifetime.start
//
// WIN64-NOT: lifetime.start
// WIN64: call void @handle_exception
// WIN64-SAME: (ptr noundef %[[e_addr]])
// WIN64-NOT: lifetime.end
// WIN64: catchret

extern "C" void catch_int_unnamed() {
  try {
    might_throw();
  } catch (int) {
  }
}

// WIN64-LABEL: define dso_local void @catch_int_unnamed()
// WIN64: catchpad within %{{.*}} [ptr @"??_R0H@8", i32 0, ptr null]
// WIN64: catchret

struct A {
  A();
  A(const A &o);
  ~A();
  int a;
};

struct B : A {
  B();
  B(const B &o);
  ~B();
  int b;
};

extern "C" void catch_a_byval() {
  try {
    might_throw();
  } catch (A e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define dso_local void @catch_a_byval()
// WIN64: %[[e_addr:[^ ]*]] = alloca %struct.A
// WIN64: catchpad within %{{[^ ]*}} [ptr @"??_R0?AUA@@@8", i32 0, ptr %[[e_addr]]]
// WIN64: call void @handle_exception(ptr noundef %[[e_addr]])
// WIN64: catchret

extern "C" void catch_a_ref() {
  try {
    might_throw();
  } catch (A &e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define dso_local void @catch_a_ref()
// WIN64: %[[e_addr:[^ ]*]] = alloca ptr
// WIN64: catchpad within %{{[^ ]*}} [ptr @"??_R0?AUA@@@8", i32 8, ptr %[[e_addr]]]
// WIN64: %[[eptr:[^ ]*]] = load ptr, ptr %[[e_addr]]
// WIN64: call void @handle_exception(ptr noundef %[[eptr]])
// WIN64: catchret

extern "C" void fn_with_exc_spec() throw(int) {
  might_throw();
}

// WIN64-LABEL: define dso_local void @fn_with_exc_spec()
// WIN64: call void @might_throw()
// WIN64-NEXT: ret void

extern "C" void catch_nested() {
  try {
    might_throw();
  } catch (int) {
    try {
      might_throw();
    } catch (int) {
      might_throw();
    }
  }
}

// WIN64-LABEL: define dso_local void @catch_nested()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %{{.*}} unwind label %[[catchswitch_outer:[^ ]*]]
//
// WIN64: [[catchswitch_outer]]
// WIN64: %[[catchswitch_outer_scope:[^ ]*]] = catchswitch within none [label %[[catch_int_outer:[^ ]*]]] unwind to caller
//
// WIN64: [[catch_int_outer]]
// WIN64: %[[catchpad:[^ ]*]] = catchpad within %[[catchswitch_outer_scope]] [ptr @"??_R0H@8", i32 0, ptr null]
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont2:[^ ]*]] unwind label %[[catchswitch_inner:[^ ]*]]
//
// WIN64: [[catchswitch_inner]]
// WIN64: %[[catchswitch_inner_scope:[^ ]*]] = catchswitch within %[[catchpad]] [label %[[catch_int_inner:[^ ]*]]] unwind to caller
//
// WIN64: [[catch_int_inner]]
// WIN64: catchpad within %[[catchswitch_inner_scope]] [ptr @"??_R0H@8", i32 0, ptr null]
// WIN64-NEXT: call void @might_throw()
// WIN64: catchret {{.*}} to label %[[catchret2:[^ ]*]]
//
// WIN64: [[catchret2]]
// WIN64: catchret {{.*}} to label %[[mainret:[^ ]*]]
//
// WIN64: [[mainret]]
// WIN64: ret void
