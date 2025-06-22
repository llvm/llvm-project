// RUN: clang-format -style='{BasedOnStyle: LLVM, AlignEscapedNewlines: DontAlign}' %s \
// RUN:   | FileCheck -strict-whitespace %s

// CHECK: {{^#define TAG\(\.\.\.\) \\}}
// CHECK: {{^  struct a \{\};}}
// There is whitespace following  v  this backslash!
#define TAG(...)  struct a {     \     
  };

// CHECK: {{^int   i;}}
// The comment below eats its following line because of the line splice.
// I also have trailing whitespace. Nom nom nom \     
int   i;
