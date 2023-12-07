// RUN: %clang_cc1 -emit-llvm -triple wasm32-unknown-unknown %s -o - | \
// RUN: FileCheck -check-prefix CHECK-NOEXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple wasm64-unknown-unknown %s -o - | \
// RUN: FileCheck -check-prefix CHECK-NOEXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple ppc64le-unknown-linux-gnu %s -o - | \
// RUN: FileCheck -check-prefix CHECK-NOEXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple arm64-unknown-linux-gnu %s -o - | \
// RUN: FileCheck -check-prefix CHECK-EXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple arm64-apple-ios %s -o - | \
// RUN: FileCheck -check-prefix CHECK-EXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple aarch64-unknown-linux-gnu %s -o - | \
// RUN: FileCheck -check-prefix CHECK-EXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple mips-unknown-linux-gnu %s -o - | \
// RUN: FileCheck -check-prefix CHECK-EXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-fuchsia %s -o - | \
// RUN: FileCheck -check-prefix CHECK-EXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple powerpc-unknown-aix %s -o - | \
// RUN: FileCheck -check-prefix CHECK-NOEXTRAALIGN %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-unknown-windows-msvc %s -o - | \
// RUN: FileCheck -check-prefix CHECK-MSVC %s

class t {
public:
  virtual void foo(void);
  void bar(void);
  void baz(void);
};

// The original alignment is observed if >=2, regardless of any extra alignment
// of member functions.
[[gnu::aligned(16)]]
void
t::baz(void) {
// CHECK-NOEXTRAALIGN: @_ZN1t3bazEv({{.*}}) #0 align 16 {
// CHECK-EXTRAALIGN: @_ZN1t3bazEv({{.*}}) #0 align 16 {
// CHECK-MSVC: @"?baz@t@@QEAAXXZ"({{.*}}) #0 align 16 {
}

void
t::bar(void) {
// CHECK-NOEXTRAALIGN: @_ZN1t3barEv({{.*}}) #0 {
// CHECK-EXTRAALIGN: @_ZN1t3barEv({{.*}}) #0 align 2 {
// CHECK-MSVC: @"?bar@t@@QEAAXXZ"({{.*}}) #0 align 2 {
}

void
t::foo(void) {
// CHECK-NOEXTRAALIGN: @_ZN1t3fooEv({{.*}}) unnamed_addr #0 {
// CHECK-EXTRAALIGN: @_ZN1t3fooEv({{.*}}) unnamed_addr #0 align 2 {
// CHECK-MSVC: @"?foo@t@@UEAAXXZ"({{.*}}) unnamed_addr #0 align 2 {
}
