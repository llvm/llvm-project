// RUN: %clang %s -### -o %t.o 2>&1 -fsyntax-only -fobjc-convert-messages-to-runtime-calls -fno-objc-convert-messages-to-runtime-calls -target x86_64-apple-macosx10.10.0 | FileCheck  %s --check-prefix=DISABLE
// RUN: %clang %s -### -o %t.o 2>&1 -fsyntax-only -fno-objc-convert-messages-to-runtime-calls -fobjc-convert-messages-to-runtime-calls -target x86_64-apple-macosx10.10.0 | FileCheck  %s --check-prefix=ENABLE

// Check that we pass fobjc-convert-messages-to-runtime-calls only when supported, and not explicitly disabled.

// DISABLE: "-fno-objc-convert-messages-to-runtime-calls"
// ENABLE-NOT: "-fno-objc-convert-messages-to-runtime-calls"
