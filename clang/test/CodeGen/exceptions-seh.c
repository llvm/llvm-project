// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X64
// RUN: %clang_cc1 %s -triple i686-pc-win32 -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=CHECK --check-prefix=X86
// RUN: %clang_cc1 %s -triple aarch64-windows -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefixes=CHECK,ARM64
// RUN: %clang_cc1 %s -triple i686-pc-windows-gnu -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=X86-GNU
// RUN: %clang_cc1 %s -triple x86_64-pc-windows-gnu -fms-extensions -emit-llvm -o - \
// RUN:         | FileCheck %s --check-prefix=X64-GNU

void try_body(int numerator, int denominator, int *myres) {
  *myres = numerator / denominator;
}
// CHECK-LABEL: define dso_local void @try_body(i32 noundef %numerator, i32 noundef %denominator, ptr noundef %myres)
// CHECK: sdiv i32
// CHECK: store i32 %{{.*}}, ptr
// CHECK: ret void

int safe_div(int numerator, int denominator, int *res) {
  int myres = 0;
  int success = 1;
  __try {
    try_body(numerator, denominator, &myres);
  } __except (1) {
    success = -42;
  }
  *res = myres;
  return success;
}

// CHECK-LABEL: define dso_local i32 @safe_div(i32 noundef %numerator, i32 noundef %denominator, ptr noundef %res)
// X64-SAME:      personality ptr @__C_specific_handler
// ARM64-SAME:    personality ptr @__C_specific_handler
// X86-SAME:      personality ptr @_except_handler3
// CHECK: invoke void @try_body(i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr noundef %{{.*}}) #[[NOINLINE:[0-9]+]]
// CHECK:       to label %{{.*}} unwind label %[[catchpad:[^ ]*]]
//
// CHECK: [[catchpad]]
// X64: %[[padtoken:[^ ]*]] = catchpad within %{{[^ ]*}} [ptr null]
// ARM64: %[[padtoken:[^ ]*]] = catchpad within %{{[^ ]*}} [ptr null]
// X86: %[[padtoken:[^ ]*]] = catchpad within %{{[^ ]*}} [ptr @"?filt$0@0@safe_div@@"]
// CHECK-NEXT: catchret from %[[padtoken]] to label %[[except:[^ ]*]]
//
// CHECK: [[except]]
// CHECK: store i32 -42, ptr %[[success:[^ ]*]]
//
// CHECK: %[[res:[^ ]*]] = load i32, ptr %[[success]]
// CHECK: ret i32 %[[res]]

// 32-bit SEH needs this filter to save the exception code.
//
// X86-LABEL: define internal i32 @"?filt$0@0@safe_div@@"()
// X86: %[[ebp:[^ ]*]] = call ptr @llvm.frameaddress.p0(i32 1)
// X86: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @safe_div, ptr %[[ebp]])
// X86: call ptr @llvm.localrecover(ptr @safe_div, ptr %[[fp]], i32 0)
// X86: load ptr, ptr
// X86: load ptr, ptr
// X86: load i32, ptr
// X86: store i32 %{{.*}}, ptr
// X86: ret i32 1

// Mingw uses msvcrt, so it can also use _except_handler3.
// X86-GNU-LABEL: define dso_local i32 @safe_div(i32 noundef %numerator, i32 noundef %denominator, ptr noundef %res)
// X86-GNU-SAME:      personality ptr @_except_handler3
// X64-GNU-LABEL: define dso_local i32 @safe_div(i32 noundef %numerator, i32 noundef %denominator, ptr noundef %res)
// X64-GNU-SAME:      personality ptr @__C_specific_handler

void j(void);

int filter_expr_capture(void) {
  int r = 42;
  __try {
    j();
  } __except(r = -1) {
    r = 13;
  }
  return r;
}

// CHECK-LABEL: define dso_local i32 @filter_expr_capture()
// X64-SAME: personality ptr @__C_specific_handler
// ARM64-SAME: personality ptr @__C_specific_handler
// X86-SAME: personality ptr @_except_handler3
// X64: call void (...) @llvm.localescape(ptr %[[r:[^ ,]*]])
// ARM64: call void (...) @llvm.localescape(ptr %[[r:[^ ,]*]])
// X86: call void (...) @llvm.localescape(ptr %[[r:[^ ,]*]], ptr %[[code:[^ ,]*]])
// CHECK: store i32 42, ptr %[[r]]
// CHECK: invoke void @j() #[[NOINLINE]]
//
// CHECK: catchpad within %{{[^ ]*}} [ptr @"?filt$0@0@filter_expr_capture@@"]
// CHECK: store i32 13, ptr %[[r]]
//
// CHECK: %[[rv:[^ ]*]] = load i32, ptr %[[r]]
// CHECK: ret i32 %[[rv]]

// X64-LABEL: define internal i32 @"?filt$0@0@filter_expr_capture@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// X64: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @filter_expr_capture, ptr %frame_pointer)
// X64: call ptr @llvm.localrecover(ptr @filter_expr_capture, ptr %[[fp]], i32 0)
//
// ARM64-LABEL: define internal i32 @"?filt$0@0@filter_expr_capture@@"(ptr noundef %exception_pointers, ptr noundef %frame_pointer)
// ARM64: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @filter_expr_capture, ptr %frame_pointer)
// ARM64: call ptr @llvm.localrecover(ptr @filter_expr_capture, ptr %[[fp]], i32 0)
//
// X86-LABEL: define internal i32 @"?filt$0@0@filter_expr_capture@@"()
// X86: %[[ebp:[^ ]*]] = call ptr @llvm.frameaddress.p0(i32 1)
// X86: %[[fp:[^ ]*]] = call ptr @llvm.eh.recoverfp(ptr @filter_expr_capture, ptr %[[ebp]])
// X86: call ptr @llvm.localrecover(ptr @filter_expr_capture, ptr %[[fp]], i32 0)
//
// CHECK: store i32 -1, ptr %{{.*}}
// CHECK: ret i32 -1

int nested_try(void) {
  int r = 42;
  __try {
    __try {
      j();
      r = 0;
    } __except(_exception_code() == 123) {
      r = 123;
    }
  } __except(_exception_code() == 456) {
    r = 456;
  }
  return r;
}
// CHECK-LABEL: define dso_local i32 @nested_try()
// X64-SAME: personality ptr @__C_specific_handler
// ARM64-SAME: personality ptr @__C_specific_handler
// X86-SAME: personality ptr @_except_handler3
// CHECK: store i32 42, ptr %[[r:[^ ,]*]]
// CHECK: invoke void @j() #[[NOINLINE]]
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[cswitch_inner:[^ ]*]]
//
// CHECK: [[cswitch_inner]]
// CHECK: %[[cs_inner:[^ ]*]] = catchswitch within none [label %[[cpad_inner:[^ ]*]]] unwind label %[[cswitch_outer:[^ ]*]]
//
// CHECK: [[cswitch_outer]]
// CHECK: %[[cs_outer:[^ ]*]] = catchswitch within none [label %[[cpad_outer:[^ ]*]]] unwind to caller
//
// CHECK: [[cpad_outer]]
// CHECK: catchpad within %{{[^ ]*}} [ptr @"?filt$0@0@nested_try@@"]
// CHECK-NEXT: catchret {{.*}} to label %[[except_outer:[^ ]*]]
//
// CHECK: [[except_outer]]
// CHECK: store i32 456, ptr %[[r]]
// CHECK: br label %[[outer_try_cont:[^ ]*]]
//
// CHECK: [[outer_try_cont]]
// CHECK: %[[r_load:[^ ]*]] = load i32, ptr %[[r]]
// CHECK: ret i32 %[[r_load]]
//
// CHECK: [[cpad_inner]]
// CHECK: catchpad within %[[cs_inner]] [ptr @"?filt$1@0@nested_try@@"]
// CHECK-NEXT: catchret {{.*}} to label %[[except_inner:[^ ]*]]
//
// CHECK: [[except_inner]]
// CHECK: store i32 123, ptr %[[r]]
// CHECK: br label %[[inner_try_cont:[^ ]*]]
//
// CHECK: [[inner_try_cont]]
// CHECK: br label %[[outer_try_cont]]
//
// CHECK: [[cont]]
// CHECK: store i32 0, ptr %[[r]]
// CHECK: br label %[[inner_try_cont]]
//
// CHECK-LABEL: define internal i32 @"?filt$0@0@nested_try@@"({{.*}})
// X86: call ptr @llvm.eh.recoverfp({{.*}})
// CHECK: load ptr, ptr
// CHECK: load i32, ptr
// CHECK: icmp eq i32 %{{.*}}, 456
//
// CHECK-LABEL: define internal i32 @"?filt$1@0@nested_try@@"({{.*}})
// X86: call ptr @llvm.eh.recoverfp({{.*}})
// CHECK: load ptr, ptr
// CHECK: load i32, ptr
// CHECK: icmp eq i32 %{{.*}}, 123

int basic_finally(int g) {
  __try {
    j();
  } __finally {
    ++g;
  }
  return g;
}
// CHECK-LABEL: define dso_local i32 @basic_finally(i32 noundef %g)
// X64-SAME: personality ptr @__C_specific_handler
// ARM64-SAME: personality ptr @__C_specific_handler
// X86-SAME: personality ptr @_except_handler3
// CHECK: %[[g_addr:[^ ]*]] = alloca i32, align 4
// CHECK: call void (...) @llvm.localescape(ptr %[[g_addr]])
// CHECK: store i32 %g, ptr %[[g_addr]]
//
// CHECK: invoke void @j()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[cleanuppad:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call void @"?fin$0@0@basic_finally@@"({{i8 noundef( zeroext)?}} 0, ptr noundef %[[fp]])
// CHECK: load i32, ptr %[[g_addr]], align 4
// CHECK: ret i32
//
// CHECK: [[cleanuppad]]
// CHECK: %[[padtoken:[^ ]*]] = cleanuppad within none []
// CHECK: %[[fp:[^ ]*]] = call ptr @llvm.localaddress()
// CHECK: call void @"?fin$0@0@basic_finally@@"({{i8 noundef( zeroext)?}} 1, ptr noundef %[[fp]])
// CHECK: cleanupret from %[[padtoken]] unwind to caller

// CHECK: define internal void @"?fin$0@0@basic_finally@@"({{i8 noundef( zeroext)?}} %abnormal_termination, ptr noundef %frame_pointer)
// CHECK:   call ptr @llvm.localrecover(ptr @basic_finally, ptr %frame_pointer, i32 0)
// CHECK:   load i32, ptr %{{.*}}, align 4
// CHECK:   add nsw i32 %{{.*}}, 1
// CHECK:   store i32 %{{.*}}, ptr %{{.*}}, align 4
// CHECK:   ret void

int returns_int(void);
int except_return(void) {
  __try {
    return returns_int();
  } __except(1) {
    return 42;
  }
}
// CHECK-LABEL: define dso_local i32 @except_return()
// CHECK: %[[tmp:[^ ]*]] = invoke i32 @returns_int()
// CHECK:       to label %[[cont:[^ ]*]] unwind label %[[catchpad:[^ ]*]]
//
// CHECK: [[catchpad]]
// CHECK: catchpad
// CHECK: catchret
// CHECK: store i32 42, ptr %[[rv:[^ ]*]]
// CHECK: br label %[[retbb:[^ ]*]]
//
// CHECK: [[cont]]
// CHECK: store i32 %[[tmp]], ptr %[[rv]]
// CHECK: br label %[[retbb]]
//
// CHECK: [[retbb]]
// CHECK: %[[r:[^ ]*]] = load i32, ptr %[[rv]]
// CHECK: ret i32 %[[r]]


// PR 24751: don't assert if a variable is used twice in a __finally block.
// Also, make sure we don't do redundant work to capture/project it.
void finally_capture_twice(int x) {
  __try {
  } __finally {
    int y = x;
    int z = x;
  }
}
//
// CHECK-LABEL: define dso_local void @finally_capture_twice(
// CHECK:         [[X:%.*]] = alloca i32, align 4
// CHECK:         call void (...) @llvm.localescape(ptr [[X]])
// CHECK-NEXT:    store i32 {{.*}}, ptr [[X]], align 4
// CHECK-NEXT:    [[LOCAL:%.*]] = call ptr @llvm.localaddress()
// CHECK-NEXT:    call void [[FINALLY:@.*]](i8 noundef{{ zeroext | }}0, ptr noundef [[LOCAL]])
// CHECK:       define internal void [[FINALLY]](
// CHECK:         [[LOCAL:%.*]] = call ptr @llvm.localrecover(
// CHECK-NEXT:    [[Y:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[Z:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store ptr
// CHECK-NEXT:    store i8
// CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[LOCAL]], align 4
// CHECK-NEXT:    store i32 [[T0]], ptr [[Y]], align 4
// CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[LOCAL]], align 4
// CHECK-NEXT:    store i32 [[T0]], ptr [[Z]], align 4
// CHECK-NEXT:    ret void

int exception_code_in_except(void) {
  __try {
    try_body(0, 0, 0);
  } __except(1) {
    return _exception_code();
  }
  return 0;
}

// CHECK-LABEL: define dso_local i32 @exception_code_in_except()
// CHECK: %[[ret_slot:[^ ]*]] = alloca i32
// CHECK: %[[code_slot:[^ ]*]] = alloca i32
// CHECK: invoke void @try_body(i32 noundef 0, i32 noundef 0, ptr noundef null)
// CHECK: %[[pad:[^ ]*]] = catchpad
// CHECK: catchret from %[[pad]]
// X64: %[[code:[^ ]*]] = call i32 @llvm.eh.exceptioncode(token %[[pad]])
// X64: store i32 %[[code]], ptr %[[code_slot]]
// ARM64: %[[code:[^ ]*]] = call i32 @llvm.eh.exceptioncode(token %[[pad]])
// ARM64: store i32 %[[code]], ptr %[[code_slot]]
// CHECK: %[[ret1:[^ ]*]] = load i32, ptr %[[code_slot]]
// CHECK: store i32 %[[ret1]], ptr %[[ret_slot]]
// CHECK: %[[ret2:[^ ]*]] = load i32, ptr %[[ret_slot]]
// CHECK: ret i32 %[[ret2]]

// CHECK: attributes #[[NOINLINE]] = { {{.*noinline.*}} }
