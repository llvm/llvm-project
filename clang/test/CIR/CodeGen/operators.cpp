// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class __attribute__((__visibility__("default"))) exception_ptr
{
    void* __ptr_;
public:
    explicit operator bool() const noexcept {return __ptr_ != nullptr;}
};

// TODO: for now only check that this doesn't crash, in the future check operator
// bool codegen.

// CHECK: module