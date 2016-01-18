// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc \
// RUN:     -mconstructor-aliases -fexceptions -fcxx-exceptions -fnew-ms-eh \
// RUN:     -O1 -disable-llvm-optzns \
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

// WIN64-LABEL: define void @catch_all()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// WIN64: [[lpad]]
// WIN64: catchpad [i8* null, i32 64, i8* null]
// WIN64:     to label %[[catchit:[^ ]*]] unwind label %{{.*}}
//
// WIN64: [[catchit]]
// WIN64: call void @recover()
// WIN64: catchret %{{.*}} to label %[[catchret:[^ ]*]]
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

// WIN64-LABEL: define void @catch_int()
// WIN64: catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i32* %[[e_addr:[^\]]*]]]
//
// The catchpad instruction starts the lifetime of 'e'. Unfortunately, that
// leaves us with nowhere to put lifetime.start, so we don't emit lifetime
// markers for now.
// WIN64-NOT: lifetime.start
//
// WIN64: %[[e_i8:[^ ]*]] = bitcast i32* %[[e_addr]] to i8*
// WIN64-NOT: lifetime.start
// WIN64: call void @handle_exception
// WIN64-SAME: (i8* %[[e_i8]])
// WIN64-NOT: lifetime.end
// WIN64: catchret

extern "C" void catch_int_unnamed() {
  try {
    might_throw();
  } catch (int) {
  }
}

// WIN64-LABEL: define void @catch_int_unnamed()
// WIN64: catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
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

// WIN64-LABEL: define void @catch_a_byval()
// WIN64: %[[e_addr:[^ ]*]] = alloca %struct.A
// WIN64: catchpad [%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8", i32 0, %struct.A* %[[e_addr]]]
// WIN64: %[[e_i8:[^ ]*]] = bitcast %struct.A* %[[e_addr]] to i8*
// WIN64: call void @handle_exception(i8* %[[e_i8]])
// WIN64: catchret

extern "C" void catch_a_ref() {
  try {
    might_throw();
  } catch (A &e) {
    handle_exception(&e);
  }
}

// WIN64-LABEL: define void @catch_a_ref()
// WIN64: %[[e_addr:[^ ]*]] = alloca %struct.A*
// WIN64: catchpad [%rtti.TypeDescriptor7* @"\01??_R0?AUA@@@8", i32 8, %struct.A** %[[e_addr]]]
// WIN64: %[[eptr:[^ ]*]] = load %struct.A*, %struct.A** %[[e_addr]]
// WIN64: %[[eptr_i8:[^ ]*]] = bitcast %struct.A* %[[eptr]] to i8*
// WIN64: call void @handle_exception(i8* %[[eptr_i8]])
// WIN64: catchret

extern "C" void fn_with_exc_spec() throw(int) {
  might_throw();
}

// WIN64-LABEL: define void @fn_with_exc_spec()
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

// WIN64-LABEL: define void @catch_nested()
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %{{.*}} unwind label %[[lp1:[^ ]*]]
//
// WIN64: [[lp1]]
// WIN64: catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
// WIN64:     to label %[[catchit:[^ ]*]] unwind label %{{.*}}
//
// WIN64: [[catchit]]
// WIN64: invoke void @might_throw()
// WIN64-NEXT: to label %[[cont2:[^ ]*]] unwind label %[[lp2:[^ ]*]]
//
// WIN64: [[lp2]]
// WIN64: catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
// WIN64:     to label %[[catchit2:[^ ]*]] unwind label %[[endcatch2:[^ ]*]]
//
// WIN64: [[catchit2]]
// WIN64-NEXT: invoke void @might_throw()
// WIN64-NEXT:    to label %[[catchret1:[^ ]*]] unwind label %[[endcatch2]]
//
// WIN64: [[catchret1]]
// WIN64: catchret {{.*}} to label %[[catchret2:[^ ]*]]
//
// WIN64: [[catchret2]]
// WIN64: catchret {{.*}} to label %[[mainret:[^ ]*]]
//
// WIN64: [[mainret]]
// WIN64: ret void
//
// WIN64: [[endcatch2]]
// WIN64: catchendpad unwind label %{{.*}}
