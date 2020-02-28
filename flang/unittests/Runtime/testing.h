#ifndef FORTRAN_TEST_RUNTIME_TESTING_H_
#define FORTRAN_TEST_RUNTIME_TESTING_H_

#include <cstddef>
#include <iosfwd>

namespace llvm {
 class raw_ostream;
}

void StartTests();
llvm::raw_ostream &Fail();
int EndTests();

void SetCharacter(char *, std::size_t, const char *);

#endif  // FORTRAN_TEST_RUNTIME_TESTING_H_
