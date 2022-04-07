// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.base-none-eabi -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-NOSE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbebv8m.base-none-eabi -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-NOSE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbv8m.base-none-eabi -mcmse -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-SE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -triple thumbebv8m.base-none-eabi -mcmse -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK-SE --check-prefix=CHECK

typedef void (*callback_t)(void) __attribute__((cmse_nonsecure_call));
typedef void callback2_t(void) __attribute__((cmse_nonsecure_call));

void f1(callback_t fptr)
{
    fptr();
}

void f2(callback2_t *fptr)
{
    fptr();
}

void f3(void) __attribute__((cmse_nonsecure_entry));
void f3(void)
{
}

void f4(void) __attribute__((cmse_nonsecure_entry))
{
}

// CHECK: define{{.*}} void @f1(void ()* nocapture noundef readonly %fptr) {{[^#]*}}#0 {
// CHECK: call void %fptr() #2
// CHECK: define{{.*}} void @f2(void ()* nocapture noundef readonly %fptr) {{[^#]*}}#0 {
// CHECK: call void %fptr() #2
// CHECK: define{{.*}} void @f3() {{[^#]*}}#1 {
// CHECK: define{{.*}} void @f4() {{[^#]*}}#1 {

// CHECK-NOSE-NOT: cmse_nonsecure_entry
// CHECK-NOSE-NOT: cmse_nonsecure_call
// CHECK-SE: attributes #0 = { nounwind
// CHECK-SE: attributes #1 = { {{.*}} "cmse_nonsecure_entry"
// CHECK-SE: attributes #2 = { {{.*}} "cmse_nonsecure_call"
