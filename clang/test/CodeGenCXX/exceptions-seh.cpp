// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -fcxx-exceptions -fexceptions | \
// RUN:         FileCheck %s --check-prefix=CHECK --check-prefix=CXXEH
// RUN: %clang_cc1 -std=c++11 -fblocks -fms-extensions %s -triple=x86_64-windows-msvc -emit-llvm \
// RUN:         -o - -mconstructor-aliases -O1 -disable-llvm-optzns | \
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

// CXXEH-LABEL: define void @use_cxx()
// CXXEH-SAME:  personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CXXEH: call %struct.HasCleanup* @"\01??0HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: invoke void @might_throw()
// CXXEH:       to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// CXXEH: [[cont]]
// CXXEH: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: ret void
//
// CXXEH: [[lpad]]
// CXXEH: cleanuppad
// CXXEH: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// CXXEH: cleanupret

// NOCXX-LABEL: define void @use_cxx()
// NOCXX-NOT: invoke
// NOCXX: call %struct.HasCleanup* @"\01??0HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: call void @might_throw()
// NOCXX-NOT: invoke
// NOCXX: call void @"\01??1HasCleanup@@QEAA@XZ"(%struct.HasCleanup* %{{.*}})
// NOCXX-NOT: invoke
// NOCXX: ret void

extern "C" void use_seh() {
  __try {
    might_throw();
  } __except(1) {
  }
}

// Make sure we use __C_specific_handler for SEH.

// CHECK-LABEL: define void @use_seh()
// CHECK-SAME:  personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
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

// CXXEH-LABEL: define void @"\01?use_seh_in_lambda@@YAXXZ"()
// CXXEH-SAME:  personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
// CXXEH: cleanuppad

// NOCXX-LABEL: define void @"\01?use_seh_in_lambda@@YAXXZ"()
// NOCXX-NOT: invoke
// NOCXX: ret void

// CHECK-LABEL: define internal void @"\01??R<lambda_0>@?0??use_seh_in_lambda@@YAXXZ@QEBA@XZ"(%class.anon* %this)
// CXXEH-SAME:  personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
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

// CHECK-LABEL: define linkonce_odr void @use_seh_in_inline_func() #{{[0-9]+}} comdat
// CHECK-SAME:  personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*)
// CHECK: invoke void @might_throw()
//
// CHECK: catchpad {{.*}} [i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@use_seh_in_inline_func@@" to i8*)]
//
// CHECK: invoke void @might_throw()
//
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"\01?fin$0@0@use_seh_in_inline_func@@"(i8 0, i8* %[[fp]])
// CHECK: ret void
//
// CHECK: cleanuppad
// CHECK: %[[fp:[^ ]*]] = call i8* @llvm.localaddress()
// CHECK: call void @"\01?fin$0@0@use_seh_in_inline_func@@"(i8 1, i8* %[[fp]])

// CHECK-LABEL: define internal i32 @"\01?filt$0@0@use_seh_in_inline_func@@"(i8* %exception_pointers, i8* %frame_pointer) #{{[0-9]+}} comdat($use_seh_in_inline_func)
// CHECK: icmp eq i32 %{{.*}}, 424242
// CHECK: zext i1 %{{.*}} to i32
// CHECK: ret i32

// CHECK-LABEL: define internal void @"\01?fin$0@0@use_seh_in_inline_func@@"(i8 %abnormal_termination, i8* %frame_pointer) #{{[0-9]+}} comdat($use_seh_in_inline_func)
// CHECK: store i32 1234, i32* @my_unique_global

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
