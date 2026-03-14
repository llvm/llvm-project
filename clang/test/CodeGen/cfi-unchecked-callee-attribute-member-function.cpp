// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize=cfi-mfcall -o - %s -fvisibility=hidden | FileCheck %s

#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

class A {};

// CHECK-LABEL: _Z14MemberFuncCallP1AMS_FvvE
void MemberFuncCall(A *s, void (A::*p)()) {
  // CHECK:      memptr.virtual:
  // CHECK-NEXT:   [[VTABLE:%.*]] = load ptr, ptr {{.*}}, align 8
  // CHECK-NEXT:   [[OFFSET:%.*]] = sub i64 %memptr.ptr, 1
  // CHECK-NEXT:   [[FUNC:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 [[OFFSET]]
  // CHECK-NEXT:   [[VALID:%.*]] = call i1 @llvm.type.test(ptr [[FUNC]], metadata !"_ZTSM1AFvvE.virtual")
  // CHECK-NEXT:   [[FUNC:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 [[OFFSET]]
  // CHECK-NEXT:   %memptr.virtualfn = load ptr, ptr [[FUNC]], align 8
  // CHECK-NEXT:   {{.*}}= call i1 @llvm.type.test(ptr [[VTABLE]], metadata !"all-vtables")
  // CHECK-NEXT:   br i1 [[VALID]], label %[[CONT:.*]], label %handler.cfi_check_fail{{.*}}

  // CHECK:      [[CONT]]:
  // CHECK-NEXT:   br label %memptr.end

  // CHECK:      memptr.nonvirtual:
  // CHECK-NEXT:   %memptr.nonvirtualfn = inttoptr i64 %memptr.ptr to ptr
  // CHECK-NEXT:   [[VALID:%.*]] = call i1 @llvm.type.test(ptr %memptr.nonvirtualfn, metadata !"_ZTSM1AFvvE")
  // CHECK-NEXT:   [[VALID2:%.*]] = or i1 false, [[VALID]]
  // CHECK-NEXT:   br i1 [[VALID2]], label %[[CONT2:.*]], label %handler.cfi_check_fail{{.*}}

  // CHECK:      [[CONT2]]:
  // CHECK-NEXT:   br label %memptr.end

  // CHECK:      memptr.end:
  // CHECK-NEXT:   {{.*}} = phi ptr [ %memptr.virtualfn, %[[CONT]] ], [ %memptr.nonvirtualfn, %[[CONT2]] ]
  (s->*p)();
}

// CHECK-LABEL: _Z19MemberFuncCallNoCFIP1AMS_FvvE
// CHECK-NOT: llvm.type.test
void MemberFuncCallNoCFI(A *s, void (CFI_UNCHECKED_CALLEE A::*p)()) {
  // CHECK:      memptr.virtual:
  // CHECK-NEXT:   [[VTABLE:%.*]] = load ptr, ptr {{.*}}, align 8
  // CHECK-NEXT:   [[OFFSET:%.*]] = sub i64 %memptr.ptr, 1
  // CHECK-NEXT:   [[FUNC:%.*]] = getelementptr i8, ptr [[VTABLE]], i64 [[OFFSET]]
  // CHECK-NEXT:   %memptr.virtualfn = load ptr, ptr [[FUNC]], align 8
  // CHECK-NEXT:   br label %memptr.end

  // CHECK:      memptr.nonvirtual:
  // CHECK-NEXT:   %memptr.nonvirtualfn = inttoptr i64 %memptr.ptr to ptr
  // CHECK-NEXT:   br label %memptr.end

  // CHECK:      memptr.end:
  // CHECK-NEXT:   {{.*}} = phi ptr [ %memptr.virtualfn, %memptr.virtual ], [ %memptr.nonvirtualfn, %memptr.nonvirtual ]
  (s->*p)();
}
