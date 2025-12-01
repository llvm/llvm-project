// RUN: %clang_cc1 -debug-info-kind=constructor -triple bpf -emit-llvm %s -o - | FileCheck %s

class Foo {
public:
  virtual ~Foo() noexcept;
};

class Bar : public Foo {
public:
  Bar() noexcept {}
  ~Bar() noexcept override;
};

// CHECK: declare !dbg !{{[0-9]+}} void @_ZN3FooD2Ev(ptr dead_on_return noundef nonnull align 8 dereferenceable(8))
