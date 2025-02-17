// REQUIRES: x86-registered-target

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -triple x86_64-unknown-fuchsia -O2 -flto -ffat-lto-objects \
// RUN:          -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fvisibility=hidden \
// RUN:          -emit-llvm -o - %t/a.cpp \
// RUN:   | FileCheck %s --check-prefix=TYPE_TEST

//--- a.cpp
// TYPE_TEST: llvm.embedded.object
// TYPE_TEST-SAME: section ".llvm.lto"

// COM: The FatLTO pipeline should remove all llvm.type.test instructions.
// TYPE_TEST-LABEL: define hidden void @foo
// TYPE_TEST-NOT:   @llvm.type.test
// TYPE_TEST-NEXT:  entry:
// TYPE_TEST-NEXT:    %cmp14.not = icmp eq i64 %len, 0
// TYPE_TEST-NEXT:    br i1 %cmp14.not, label %for.end7, label %for.cond1.preheader.preheader
// TYPE_TEST-LABEL: for.cond1.preheader.preheader:                    ; preds = %entry
// TYPE_TEST-NEXT:    %arrayidx.1 = getelementptr inbounds nuw i8, ptr %ptr, i64 4
// TYPE_TEST-NEXT:    br label %for.cond1.preheader

// The code below is a reduced case from https://github.com/llvm/llvm-project/issues/112053
#define __PRINTFLIKE(__fmt, __varargs) __attribute__((__format__(__printf__, __fmt, __varargs)))
typedef void func(void* arg, const char* fmt, ...) __PRINTFLIKE(2, 3);
typedef __SIZE_TYPE__ size_t;
typedef unsigned long uintptr_t;

extern "C"
void foo(const void* ptr, size_t len, long disp_addr,
                     func* printf_func, void* printf_arg) {
  uintptr_t address = (uintptr_t)ptr;
  size_t count;

  for (count = 0; count < len; count += 16) {
    union {
      unsigned int buf[4];
      unsigned char cbuf[16];
    } u;
    size_t s = 10;
    size_t i;

    for (i = 0; i < s / 4; i++) {
      u.buf[i] = ((const unsigned int*)address)[i];
      printf_func(printf_arg, "%08x ", static_cast<unsigned int>(u.buf[i]));
    }
  }
}

//--- b.cpp
// COM: Prior to the introduction of the FatLTO cleanup pass, this used to cause
// COM: the backend to crash, either due to an assertion failure, or because
// COM: the CFI instructions couldn't be correctly generated. So, check to make
// COM: sure that the FatLTO pipeline used by clang does not regress.

// COM: Check the generated IR doesn't contain llvm.type.checked.load in the final IR.
// RUN: %clang_cc1 -triple=x86_64-unknown-fuchsia -O1 -emit-llvm -o - \
// RUN:      -ffat-lto-objects -fvisibility=hidden \
// RUN:      -fno-rtti -fsanitize=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fsanitize-trap=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fwhole-program-vtables %t/b.cpp 2>&1 | FileCheck %s --check-prefix=NO_CHECKED_LOAD

// RUN: %clang_cc1 -triple=x86_64-unknown-fuchsia -O1 -emit-llvm -o - \
// RUN:      -ffat-lto-objects -fvisibility=hidden -fexperimental-relative-c++-abi-vtables \
// RUN:      -fno-rtti -fsanitize=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fsanitize-trap=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fwhole-program-vtables %t/b.cpp 2>&1 | FileCheck %s --check-prefix=NO_CHECKED_LOAD

// COM: Note that the embedded bitcode section will contain references to
// COM: llvm.type.checked.load, so we need to match the function body first.
// NO_CHECKED_LOAD-LABEL: entry:
// NO_CHECKED_LOAD-NEXT:    %vtable = load ptr, ptr %p1
// NO_CHECKED_LOAD-NOT:   llvm.type.checked.load
// NO_CHECKED_LOAD-NEXT:    %vfunc = load ptr, ptr %vtable
// NO_CHECKED_LOAD-NEXT:    %call = tail call {{.*}} %vfunc(ptr {{.*}} %p1)
// NO_CHECKED_LOAD-NEXT:    ret void

// COM: Ensure that we don't crash in the backend anymore when clang uses
// COM: CFI checks with -ffat-lto-objects.
// RUN: %clang_cc1 -triple=x86_64-unknown-fuchsia -O1 -emit-codegen-only \
// RUN:      -ffat-lto-objects -fvisibility=hidden \
// RUN:      -fno-rtti -fsanitize=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fsanitize-trap=cfi-icall,cfi-mfcall,cfi-nvcall,cfi-vcall \
// RUN:      -fwhole-program-vtables %t/b.cpp

class a {
public:
  virtual long b();
};
void c(a &p1) { p1.b(); }
