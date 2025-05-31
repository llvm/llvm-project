// RUN: %clang_cc1 -ferror-limit 2 -fsyntax-only -verify %s

// expected-error@* {{too many errors emitted}}

namespace llvm {
namespace Hexagon {}
}
void set() {
  Hexagon::NoRegister;
  // expected-error@-1 {{use of undeclared identifier}}
  // expected-note@-5 {{declared here}}
  // expected-error@-3 {{no member named 'NoRegister' in namespace}}
}
template <class> struct pair { pair(int, int); };
struct HexagonMCChecker {
  static pair<int> Unconditional;
  void checkRegisters();
};
pair<int> HexagonMCChecker::Unconditional(Hexagon::NoRegister, 0);
void HexagonMCChecker::checkRegisters() {}
