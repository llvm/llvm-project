// RUN: c-index-test -index-file %s -target i686-pc-linux \
// RUN:  | FileCheck %s -check-prefix CHECK -check-prefix CHECK-LINUX
// RUN: c-index-test -index-file -Wno-unsupported-visibility %s -target i386-darwin \
// RUN:  | FileCheck %s -check-prefix CHECK -check-prefix CHECK-DARWIN

void __attribute__ (( visibility("default") )) default_visibility();
// CHECK:      <attribute>: attribute(visibility)=default
void __attribute__ (( visibility("hidden") )) hidden_visibility();
// CHECK:      <attribute>: attribute(visibility)=hidden
void __attribute__ (( visibility("protected") )) protected_visibility();
// CHECK-LINUX:      <attribute>: attribute(visibility)=protected
// CHECK-DARWIN:      <attribute>: attribute(visibility)=default

