#ifndef FORTRAN_TEST_RUNTIME_TESTING_H_
#define FORTRAN_TEST_RUNTIME_TESTING_H_

#include <cstddef>
#include <iosfwd>

void StartTests();
std::ostream &Fail();
int EndTests();

void SetCharacter(char *, std::size_t, const char *);

#endif  // FORTRAN_TEST_RUNTIME_TESTING_H_
