// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/a.cppm \
// RUN:     -emit-obj -fmodules-reduced-bmi -fmodule-output=%t/a.pcm -o %t/a.o
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/b.cppm \
// RUN:     -emit-obj -fmodules-reduced-bmi -fmodule-output=%t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm -o %t/b.o
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 %t/main.cpp \
// RUN:     -emit-obj -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm -o %t/main.o

// In the Microsoft ABI the deleting destructor is emitted together with the
// vtable, so it needs the 'operator delete' that Sema resolved for the virtual
// destructor.  Here 'operator delete' is resolved for a destructor declaration
// that is owned by module 'a' (the 'extern template' declaration puts it in
// a.pcm), so the resolution is recorded as a DeclUpdate on an imported
// declaration.  With reduced BMIs such an update used to be kept lazily and
// then dropped, because the module never wrote a declaration reference to the
// destructor; a consumer that emits the vtable then crashed in
// CGClass.cpp's EnterDtorCleanups with a null getOperatorDelete().

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

// Instantiating this constructor marks the vtable used (ActOnFinishFunctionBody)
// and Sema then resolves 'operator delete' for the imported destructor.
export void make() { new Base<int>; }

//--- main.cpp
import a;
import b;

struct Local : Base<int> {};
Local l;

int main() { return 0; }
