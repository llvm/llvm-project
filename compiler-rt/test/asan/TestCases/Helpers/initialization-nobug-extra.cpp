// Linker initialized:
int getAB();
static int ab = getAB();
// Function local statics:
int countCalls();
static int one = countCalls();
// Trivial constructor, non-trivial destructor:
int getStructWithDtorValue();
static int val = getStructWithDtorValue();
