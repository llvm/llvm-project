// RUN: %clang --target=x86_64-pc-windows-msvc -fno-rtti -### %s 2>&1 | FileCheck %s --check-prefix=NO-RTTI
// RUN: %clang --target=x86_64-pc-windows-msvc -frtti -### %s 2>&1 | FileCheck %s --check-prefix=RTTI

// RTTI-NOT: -D_HAS_STATIC_RTTI=0
// NO-RTTI: -D_HAS_STATIC_RTTI=0
