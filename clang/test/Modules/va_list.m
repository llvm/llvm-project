// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodules-ignore-macro=PREFIX -DPREFIX -I %S/Inputs/va_list \
// RUN:     -x objective-c-header %s -o %t.pch -emit-pch

// Include the pch, as a basic correctness check.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodules-ignore-macro=PREFIX -I %S/Inputs/va_list -include-pch %t.pch \
// RUN:     -x objective-c %s -fsyntax-only

// Repeat the previous emit-pch, but not we will have a global module index.
// For some reason, this results in an identifier for __va_list_tag being
// emitted into the pch.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodules-ignore-macro=PREFIX -DPREFIX -I %S/Inputs/va_list \
// RUN:     -x objective-c-header %s -o %t.pch -emit-pch

// Include the pch, which now has __va_list_tag in it, which needs to be merged.
// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -fmodules-ignore-macro=PREFIX -I %S/Inputs/va_list -include-pch %t.pch \
// RUN:     -x objective-c %s -fsyntax-only

// rdar://18039719

#ifdef PREFIX
@import va_list_b;
#endif
