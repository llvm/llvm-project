// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s -DUSE_SHIM
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// expected-no-diagnostics

// Verify both spellings: the new <swift/bridging.h> and the legacy
// extensionless <swift/bridging> shim that forwards to it.
#ifdef USE_SHIM
#include <swift/bridging>
#else
#include <swift/bridging.h>
#endif

struct LoggerSingleton { int x; } SWIFT_IMMORTAL_REFERENCE;

int getX(void) SWIFT_COMPUTED_PROPERTY;
void doThing(void *p) SWIFT_NAME(doThing(_:)) SWIFT_UNSAFE;

template <class T>
class Conforming {} SWIFT_CONFORMS_TO_PROTOCOL(SwiftModule.ProtocolName);
