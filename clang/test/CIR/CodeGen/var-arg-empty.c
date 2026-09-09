// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

struct Empty {};

struct Empty only_empty(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Empty res = __builtin_va_arg(args, struct Empty);
  __builtin_va_end(args);
  return res;
}

// An empty record travels in no register and no stack slot, so fetching one
// reads no argument and advances no field of the cursor.
// CIR-LABEL: cir.func {{.*}} @only_empty(
// CIR-NOT:     cir.va_arg
// CIR-NOT:     gp_offset
// CIR-NOT:     fp_offset
// CIR-NOT:     overflow_arg_area
// CIR:       cir.va_end

// LLVM-LABEL: define dso_local void @only_empty(i32 noundef %{{.*}}, ...)
// LLVM-NOT:     va_arg
// LLVM-NOT:     getelementptr inbounds nuw %struct.__va_list_tag
// LLVM:       call void @llvm.va_end.p0(

int empty_then_int(int count, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, count);
  struct Empty e = __builtin_va_arg(args, struct Empty);
  int i = __builtin_va_arg(args, int);
  __builtin_va_end(args);
  return i;
}

// The empty fetch leaves the cursor alone, so the int fetch that follows is
// the only one that touches gp_offset.
// CIR-LABEL: cir.func {{.*}} @empty_then_int(
// CIR:         %[[GP_OFFSET_P:.+]] = cir.get_member %{{.+}}[0] {name = "gp_offset"} : !cir.ptr<!rec___va_list_tag> -> !cir.ptr<!u32i>
// CIR:         %[[GP_OFFSET:.+]] = cir.load %[[GP_OFFSET_P]] : !cir.ptr<!u32i>, !u32i
// CIR:         %[[GP_LIMIT:.+]] = cir.const #cir.int<40> : !u32i
// CIR:         cir.cmp le %[[GP_OFFSET]], %[[GP_LIMIT]] : !u32i
// CIR-NOT:     {name = "gp_offset"}
// CIR:         cir.va_end

// LLVM-LABEL: define dso_local i32 @empty_then_int(i32 noundef %{{.*}}, ...)
// LLVM:         %[[GP_OFFSET_P:.+]] = getelementptr inbounds nuw %struct.__va_list_tag, ptr %{{.*}}, i32 0, i32 0
// LLVM:         %[[GP_OFFSET:.+]] = load i32, ptr %[[GP_OFFSET_P]]
// LLVM:         icmp ule i32 %[[GP_OFFSET]], 40
// LLVM:       call void @llvm.va_end.p0(
