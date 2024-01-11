[[noreturn]] void throw_int() {
  throw int();
}

void throw_int_wrapper() {
  [[clang::musttail]] return throw_int(); // expected-error {{'musttail' attribute may not be used with no-return-attribute functions}}
}
