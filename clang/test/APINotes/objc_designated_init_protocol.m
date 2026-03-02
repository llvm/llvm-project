// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -Wno-private-module -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s

// Regression test for APINotesWriter::addObjCMethod asserting when a protocol
// method is annotated with DesignatedInit: true in an .apinotes file.
// The writer was reconstructing the ContextTableKey with a hardcoded
// ContextKind::ObjCClass, causing the Contexts lookup to fail when the
// context was actually a protocol. Importing the module is sufficient to
// trigger the writer path that previously crashed.

// expected-no-diagnostics

#import <SomeKit/SomeKit.h>

id<InitializableProtocolDUMP> useProtocol(id<InitializableProtocolDUMP> p) {
  return [p initWithValue:0];
}
