// RUN: %clang -undef -x assembler-with-cpp -E %s
#ifndef __ASSEMBLER__
#error "Must be preprocessed as assembler."
#endif
