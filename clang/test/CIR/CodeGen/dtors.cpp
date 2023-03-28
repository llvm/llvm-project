// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class __attribute__((__visibility__("default"))) exception
{
public:
    __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) exception() noexcept {}
    __attribute__((__visibility__("hidden"))) __attribute__((__exclude_from_explicit_instantiation__)) exception(const exception&) noexcept = default;

    virtual ~exception() noexcept;
    virtual const char* what() const noexcept;
};

class __attribute__((__visibility__("default"))) bad_function_call
    : public exception
{
public:
    virtual ~bad_function_call() noexcept {}
};

// TODO: for now only check that this doesn't crash, more support soon.

// CHECK: module