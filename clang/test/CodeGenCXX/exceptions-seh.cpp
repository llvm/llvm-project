// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=CXXEH
// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -O1 -disable-llvm-passes | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=NOCXX

extern "C" unsigned long _exception_code();
extern "C" void might_throw();

struct HasCleanup {
  HasCleanup();
  ~HasCleanup();
  int padding;
};

extern "C" void use_cxx() {
  HasCleanup x;
  might_throw();
}

// Make sure we use __CxxFrameHandler3 for C++ EH.

// CXXEH-LABEL: define dso_local void @use_cxx()
// CXXEH-SAME:  personality ptr @__CxxFrameHandler3
// CXXEH: call noundef ptr @"??0HasCleanup@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// CXXEH: invoke void @might_throw()
// CXXEH:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CXXEH: [[cont]]
// CXXEH: call void @"??1HasCleanup@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// CXXEH: ret void
//
// CXXEH: [[lpad]]
// CXXEH: cleanuppad
// CXXEH: call void @"??1HasCleanup@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// CXXEH: cleanupret

// NOCXX-LABEL: define dso_local void @use_cxx()
// NOCXX-NOT: invoke
// NOCXX: call noundef ptr @"??0HasCleanup@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: call void @might_throw()
// NOCXX-NOT: invoke
// NOCXX: call void @"??1HasCleanup@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: ret void

extern "C" void use_seh() {
  __try {
    might_throw();
  } __except(1) {
  }
}

// Make sure we use __C_specific_handler for SEH.

// CHECK-LABEL: define dso_local void @use_seh()
// CHECK-SAME:  personality ptr @__C_specific_handler
// CHECK: invoke void @might_throw() #[[NOINLINE:[0-9]+]]
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CHECK: [[lpad]]
// CHECK-NEXT: %[[switch:.*]] = catchswitch within none [label %[[cpad:.*]]] unwind to caller
//
// CHECK: [[cpad]]
// CHECK-NEXT: catchpad within %[[switch]]
// CHECK: catchret {{.*}} label %[[except:[^ ]*]]
//
// CHECK: [[except]]
// CHECK: br label %[[ret:[^ ]*]]
//
// CHECK: [[ret]]
// CHECK: ret void
//
// CHECK: [[cont]]
// CHECK: br label %[[ret]]

extern "C" void nested_finally() {
  __try {
    might_throw();
  } __finally {
    __try {
      might_throw();
    } __finally {
    }
  }
}

// CHECK-LABEL: define dso_local void @nested_finally() #{{[0-9]+}}
// CHECK-SAME:  personality ptr @__C_specific_handler
// CHECK: invoke void @might_throw()
// CHECK: call void @"?fin$0@0@nested_finally@@"(i8 noundef 1, ptr {{.*}})

// CHECK-LABEL: define internal void @"?fin$0@0@nested_finally@@"
// CHECK-SAME:  personality ptr @__C_specific_handler
// CHECK: invoke void @might_throw()
// CHECK: call void @"?fin$1@0@nested_finally@@"(i8 noundef 1, ptr {{.*}})

void use_seh_in_lambda() {
  ([]() {
    __try {
      might_throw();
    } __except(1) {
    }
  })();
  HasCleanup x;
  might_throw();
}

// CXXEH-LABEL: define dso_local void @"?use_seh_in_lambda@@YAXXZ"()
// CXXEH-SAME:  personality ptr @__CxxFrameHandler3
// CXXEH: cleanuppad

// NOCXX-LABEL: define dso_local void @"?use_seh_in_lambda@@YAXXZ"()
// NOCXX-NOT: invoke
// NOCXX: ret void

// CHECK-LABEL: define internal void @"??R<lambda_0>@?0??use_seh_in_lambda@@YAXXZ@QEBA@XZ"(ptr {{[^,]*}} %this)
// CXXEH-SAME:  personality ptr @__C_specific_handler
// CHECK: invoke void @might_throw() #[[NOINLINE]]
// CHECK: catchpad

static int my_unique_global;

extern "C" inline void use_seh_in_inline_func() {
  __try {
    might_throw();
  } __except(_exception_code() == 424242) {
  }
  __try {
    might_throw();
  } __finally {
    my_unique_global = 1234;
  }
}

void use_inline() {
  use_seh_in_inline_func();
}

// CHECK-LABEL: define linkonce_odr dso_local void @use_seh_in_inline_func() #{{[0-9]+}}
// CHECK-SAME:  personality ptr @__C_specific_handler
// CHECK: invoke void @might_throw()
//
// CHECK: catchpad {{.*}} [ptr @"?filt$0@0@use_seh_in_inline_func@@"]
//
// CHECK: invoke void @might_throw()
//
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call void @"?fin$0@0@use_seh_in_inline_func@@"(i8 noundef 0, ptr noundef %[[fp]])
// CHECK: ret void
//
// CHECK: cleanuppad
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call void @"?fin$0@0@use_seh_in_inline_func@@"(i8 noundef 1, ptr noundef %[[fp]])

// CHECK-LABEL: define internal noundef i32 @"?filt$0@0@use_seh_in_inline_func@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer) #{{[0-9]+}}
// CHECK: icmp eq i32 %{{.*}}, 424242
// CHECK: zext i1 %{{.*}} to i32
// CHECK: ret i32

// CHECK-LABEL: define internal void @"?fin$0@0@use_seh_in_inline_func@@"(i8 noundef %abnormal_termination, ptr noundef %frame_pointer) #{{[0-9]+}}
// CHECK: store i32 1234, ptr @my_unique_global

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }

void seh_in_noexcept() noexcept { __try {} __finally {} }
