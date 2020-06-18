// If the linkage of the class is internal, then the stubs and proxies should
// also be internally linked.

// RUN: %clang_cc1 %s -triple=x86_64-unknown-fuchsia -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

// External linkage.
// CHECK: @_ZTI8External.rtti_proxy = hidden unnamed_addr constant { i8*, i8* }* @_ZTI8External, comdat

class External {
public:
  virtual void func();
};

void External::func() {}

// Internal linkage.
// CHECK: @_ZTIN12_GLOBAL__N_18InternalE.rtti_proxy = internal unnamed_addr constant { i8*, i8* }* @_ZTIN12_GLOBAL__N_18InternalE
namespace {

class Internal {
public:
  virtual void func();
};

void Internal::func() {}

} // namespace

// This gets the same treatment as an externally available vtable.
// CHECK: @_ZTI11LinkOnceODR.rtti_proxy = hidden unnamed_addr constant { i8*, i8* }* @_ZTI11LinkOnceODR, comdat
class LinkOnceODR {
public:
  virtual void func() {} // A method defined in the class definition results in this linkage for the vtable.
};

// Force an emission of a vtable for Internal by using it here.
void manifest_internal() {
  Internal internal;
  (void)internal;
  LinkOnceODR linkonceodr;
  (void)linkonceodr;
}

// Aliases are typically emitted after the vtable definitions but before the
// function definitions.
// CHECK: @_ZTV8External = unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV8External.local
// CHECK: @_ZTV11LinkOnceODR = linkonce_odr unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV11LinkOnceODR.local

// CHECK: define void @_ZN8External4funcEv
// CHECK: define internal void @_ZN12_GLOBAL__N_18Internal4funcEv.stub
// CHECK: define hidden void @_ZN11LinkOnceODR4funcEv.stub
