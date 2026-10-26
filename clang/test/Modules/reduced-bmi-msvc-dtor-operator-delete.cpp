// REQUIRES: x86-registered-target
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/a.cppm \
// RUN:     -emit-reduced-module-interface -fmodules-reduced-bmi -o %t/a.pcm
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/b.cppm \
// RUN:     -emit-reduced-module-interface -fmodules-reduced-bmi \
// RUN:     -fmodule-file=a=%t/a.pcm -o %t/b.pcm
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/main.cpp \
// RUN:     -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -emit-llvm -o - | FileCheck %s
//
// In the Microsoft ABI the deleting destructor is emitted together with the
// vtable, so it needs the 'operator delete' that Sema resolved for the virtual
// destructor.  Here 'operator delete' is resolved for a destructor declaration
// that is owned by module 'a' (the 'extern template' declaration puts it in
// a.pcm), so the resolution is recorded as a DeclUpdate on an imported
// declaration.  With reduced BMIs such an update used to be kept lazily and
// then dropped, because the module never wrote a declaration reference to the
// destructor; a consumer that emits the vtable then crashed in
// EnterDtorCleanups() in CGClass.cpp with a null getOperatorDelete().

// The consumer emits Base<int>'s deleting destructor, which calls the resolved
// global operator delete.
// CHECK-LABEL: define {{.*}}@"??_G?$Base@H@@UEAAPEAXI@Z"
// CHECK: call void @"??3@YAXPEAX_K@Z"

//--- a.cppm
export module a;

export template <class T> struct Base {
  __attribute__((exclude_from_explicit_instantiation)) Base() {}
  __attribute__((exclude_from_explicit_instantiation)) virtual ~Base();
};
extern template struct Base<int>;

//--- b.cppm
export module b;
import a;

// Instantiating this constructor marks the vtable used, and Sema then resolves
// 'operator delete' for the imported destructor (ActOnFinishFunctionBody).
export void make() { new Base<int>; }

//--- main.cpp
import a;
import b;

struct Local : Base<int> {};
Local l;

int main() { return 0; }
