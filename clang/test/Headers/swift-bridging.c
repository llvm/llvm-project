// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -DUSE_SHIM
// expected-no-diagnostics

// Verify both spellings: the new <swift/bridging.h> and the legacy
// extensionless <swift/bridging> shim that forwards to it.
#ifdef USE_SHIM
#include <swift/bridging>
#else
#include <swift/bridging.h>
#endif

struct LoggerSingleton { int x; } SWIFT_IMMORTAL_REFERENCE;

struct Conforming { int x; } SWIFT_CONFORMS_TO_PROTOCOL(SwiftModule.ProtocolName);

int getX(void) SWIFT_COMPUTED_PROPERTY;
void doThing(void *p) SWIFT_NAME(doThing(_:)) SWIFT_UNSAFE;
