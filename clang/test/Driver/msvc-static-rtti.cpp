// RUN: %clang -target x86_64-pc-windows-msvc -fno-rtti -### %s 2>&1 | FileCheck %s -check-prefix STATIC-RTTI-DEF
// RUN: %clang -target x86_64-pc-windows-msvc -frtti -### %s 2>&1 | FileCheck %s -check-prefix STATIC-RTTI-DEF-NOT

// STATIC-RTTI-DEF: -D_HAS_STATIC_RTTI=0
// STATIC-RTTI-DEF-NOT: -D_HAS_STATIC_RTTI=0
