// RUN: %clang_cc1 -triple thumbv7-windows -fms-extensions -verify %s

void emit_error(unsigned int opcode) {
  __emit(opcode); // expected-error {{argument to '__emit' must be a constant integer}}
}

