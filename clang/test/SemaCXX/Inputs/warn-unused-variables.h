// Verify that we don't warn about variables of internal-linkage type in
// headers, as the use may be in another TU.
namespace PR15558 {
namespace {
class A {};
}

class B {
  static A a;
  static A b;
  static const int x = sizeof(b);
};
}
