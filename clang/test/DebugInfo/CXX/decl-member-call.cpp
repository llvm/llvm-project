// RUN: %clang_cc1 -O1 -triple x86_64-unknown_unknown -emit-llvm \
// RUN:   -debug-info-kind=standalone -dwarf-version=5 %s -o - | FileCheck %s

// Ensure both nonmember and member calls to declared function
// have attached `DISubprogram`s.

int nonmember(int n);

struct S {
  int x;
  int member(int n);
};

int main(int argc, char** argv) {
  struct S s = {};
  int a = s.member(argc);
  int b = nonmember(argc);
  return a + b;
}

// CHECK: declare !dbg ![[SP1:[0-9]+]] noundef i32 @_ZN1S6memberEi(
// CHECK: declare !dbg ![[SP2:[0-9]+]] noundef i32 @_Z9nonmemberi(

// CHECK: ![[SP1]] = !DISubprogram(name: "member", linkageName: "_ZN1S6memberEi"
// CHECK: ![[SP2]] = !DISubprogram(name: "nonmember", linkageName: "_Z9nonmemberi"
