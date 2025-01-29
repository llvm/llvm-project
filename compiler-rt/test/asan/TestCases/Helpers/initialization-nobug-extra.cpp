// Linker initialized:
int getAB();
int ab = getAB();
// Function local statics:
int countCalls();
int one = countCalls();
// Trivial constructor, non-trivial destructor:
int getStructWithDtorValue();
int val = getStructWithDtorValue();
