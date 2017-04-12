#include "foo.h"
void non_modular_use_of_implicit_dtor() {
  implicit_dtor d1;
  uninst_implicit_dtor d2;
}
void use_of_instantiated_declaration_without_definition() {
  inst<int>();
}
