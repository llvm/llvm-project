// REQUIRES: host-supports-jit
// UNSUPPORTED: system-windows
//
// Emulated TLS is not supported by the LoongArch backend.
// UNSUPPORTED: target=loongarch{{.*}}
//
// An inline function that odr-uses a non-zero-initialized thread_local is
// emitted as a weak (linkonce_odr) definition into every PartialTranslationUnit
// that references it. With emulated TLS that set includes an __emutls_t.<var>
// symbol. When a later PTU re-defines the same weak set, ORC's
// IRMaterializationUnit::discard() must find each duplicated symbol in its
// SymbolToDefinition map. The emulated-TLS path used to register __emutls_t.<var>
// in SymbolFlags but not SymbolToDefinition, so discarding it dereferenced
// end() -- an assertion failure in +Asserts builds and heap corruption
// otherwise. Two PTUs each pulling in the same inline worker reproduces it.
//
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char *, ...);
template <int Tag> struct HeavyThing { static thread_local int tls; };
template <int Tag> thread_local int HeavyThing<Tag>::tls = Tag + 1;
inline int worker() { return HeavyThing<1>::tls; }
int callA() { return worker(); }
int callB() { return worker(); }
auto r = printf("tls = %d, %d\n", callA(), callB());
// CHECK: tls = 2, 2

%quit
