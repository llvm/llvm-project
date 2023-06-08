// RUN: grep -Ev "// *[A-Z-]+:" %s | clang-format --style=Google 2>&1 | FileCheck %s 
// CHECK-NOT: The new replacement overlaps with an existing replacement.

#ifdef 
    

#else 
#endif 
