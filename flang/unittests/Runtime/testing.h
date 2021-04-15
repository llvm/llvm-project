#ifndef FORTRAN_TEST_RUNTIME_TESTING_H_
#define FORTRAN_TEST_RUNTIME_TESTING_H_

#include "llvm/Support/raw_ostream.h"
#include <cstddef>

namespace llvm {
class raw_ostream;
}

void StartTests();
llvm::raw_ostream &Fail();
int EndTests();

#endif // FORTRAN_TEST_RUNTIME_TESTING_H_
