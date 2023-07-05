// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple %s -o - | \
// RUN: FileCheck -check-prefix CHECK-ITANIUM %s
// RUN: %clang_cc1 -emit-llvm -triple wasm32-unknown-unknown %s -o - | \
// RUN: FileCheck -check-prefix CHECK-WEBASSEMBLY32 %s
// RUN: %clang_cc1 -emit-llvm -triple wasm64-unknown-unknown %s -o - | \
// RUN: FileCheck -check-prefix CHECK-WEBASSEMBLY64 %s

// rdar://7268289

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
// CHECK-ITANIUM: @_ZN1t3barEv({{.*}}) #0 align 2 {
// CHECK-WEBASSEMBLY32: @_ZN1t3barEv({{.*}}) #0 {
// CHECK-WEBASSEMBLY64: @_ZN1t3barEv({{.*}}) #0 {
}

void
t::foo(void) {
// CHECK-ITANIUM: @_ZN1t3fooEv({{.*}}) unnamed_addr #0 align 2 {
// CHECK-WEBASSEMBLY32: @_ZN1t3fooEv({{.*}}) unnamed_addr #0 {
// CHECK-WEBASSEMBLY64: @_ZN1t3fooEv({{.*}}) unnamed_addr #0 {
}
