// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct Bar {
  float f1;
  float f2;
  unsigned u;
};

struct Bar varargs_aggregate(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Bar res = __builtin_va_arg(args, struct Bar);
  __builtin_va_end(args);
  return res;
}


// CIR-LABEL: cir.func {{.*}} @varargs_aggregate(
// CIR-SAME: -> !rec_anon_struct
// CIR:   %[[COERCE:.+]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_anon_struct>
// CIR:   %[[RET_ADDR:.+]] = cir.alloca "__retval" {{.*}} init : !cir.ptr<!rec_Bar>
// CIR:   %[[VAAREA:.+]] = cir.alloca "args" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[TMP_ADDR:.+]] = cir.alloca "vaarg.tmp" {{.*}} : !cir.ptr<!rec_Bar>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_ARG:.+]] = cir.va_arg %[[VA_PTR1]] : (!cir.ptr<!rec___va_list_tag>) -> !rec_Bar
// CIR:   cir.store{{.*}} %[[VA_ARG]], %[[TMP_ADDR]] : !rec_Bar, !cir.ptr<!rec_Bar>
// CIR:   cir.copy %[[TMP_ADDR]] align(4) to %[[RET_ADDR]] align(4) : !cir.ptr<!rec_Bar>
// CIR:   %[[VA_PTR2:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]] : !cir.ptr<!cir.array<!rec___va_list_tag x 1>> -> !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %[[VA_PTR2]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[RETVAL:.+]] = cir.load{{.*}} %[[RET_ADDR]] : !cir.ptr<!rec_Bar>, !rec_Bar
// CIR:   %[[SLOT:.+]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Bar>
// CIR:   cir.store %[[RETVAL]], %[[SLOT]] : !rec_Bar, !cir.ptr<!rec_Bar>
// CIR:   %[[COERCED:.+]] = cir.load %[[COERCE]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR:   cir.return %[[COERCED]] : !rec_anon_struct

// TODO(CIR): Without calling convention lowering, CIR lowers this va_arg to an
// LLVM va_arg of %struct.Bar, which the verifier rejects. Add the CIR-to-LLVM
// checks back once it is lowered the way OGCG does below.

// OGCG-LABEL: define dso_local { <2 x float>, i32 } @varargs_aggregate
// OGCG:   call void @llvm.va_start.p0(ptr %{{.*}})
// OGCG:   %[[VAARG_ADDR:.+]] = phi ptr [ %{{.*}}, %vaarg.in_reg ], [ %{{.*}}, %vaarg.in_mem ]
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %[[VAARG_ADDR]], i64 12, i1 false)

