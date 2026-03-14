int zero_init() { return 0; }
int badGlobal = zero_init();
int readBadGlobal() { return badGlobal; }

namespace badNamespace {
class BadClass {
 public:
  BadClass() { value = 0; }
  int value;
};
// Global object with non-trivial constructor.
BadClass bad_object;
}  // namespace badNamespace

int accessBadObject() { return badNamespace::bad_object.value; }
