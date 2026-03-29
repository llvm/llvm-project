// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-output=text        \
// RUN:     -analyzer-checker=core,alpha.security.ArrayBoundV2,unix.Malloc,optin.taint -verify %s

int TenElements[10];

int irrelevantAssumptions(int arg) {
  int a = TenElements[arg];
  // Here the analyzer assumes that `arg` is in bounds, but doesn't report this
  // because `arg` is not interesting for the bug.
  int b = TenElements[13];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 13, while it holds only 10 'int' elements}}
  return a + b;
}


int assumingBoth(int arg) {
  int a = TenElements[arg];
  // expected-note@-1 {{Assuming index is non-negative and less than 10, the number of 'int' elements in 'TenElements'}}
  int b = TenElements[arg]; // no additional note, we already assumed that 'arg' is in bounds
  int c = TenElements[arg + 10];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  return a + b + c;
}

int assumingBothPointerToMiddle(int arg) {
  // If we're accessing an TenElements through a pointer pointing to its middle, the checker
  // will speak about the "byte offset" measured from the beginning of the TenElements.
  int *p = TenElements + 2;
  int a = p[arg];
  // FIXME: The following note does not appear:
  //  {{Assuming byte offset is non-negative and less than 40, the extent of 'TenElements'}}
  // It seems that the analyzer "gives up" modeling this pointer arithmetics
  // and says that `p[arg]` is just an UnknownVal (instead of calculating that
  // it's equivalent to `TenElements[2+arg]`).

  int b = TenElements[arg]; // This is normal access, and only the lower bound is new.
  // expected-note@-1 {{Assuming index is non-negative}}
  int c = TenElements[arg + 10];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  return a + b + c;
}

int assumingLower(int arg) {
  // expected-note@+2 {{Assuming 'arg' is < 10}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 10)
    return 0;
  int a = TenElements[arg];
  // expected-note@-1 {{Assuming index is non-negative}}
  int b = TenElements[arg + 10];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  return a + b;
}

int assumingUpper(int arg) {
  // expected-note@+2 {{Assuming 'arg' is >= 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg < 0)
    return 0;
  int a = TenElements[arg];
  // expected-note@-1 {{Assuming index is less than 10, the number of 'int' elements in 'TenElements'}}
  int b = TenElements[arg - 10];
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset}}
  return a + b;
}

int assumingUpperIrrelevant(int arg) {
  // FIXME: The assumption "assuming index is less than 10" is printed because
  // it's assuming something about the interesting variable `arg`; however,
  // it's irrelevant because in this testcase the out of bound access is
  // deduced from the _lower_ bound on `arg`. Currently the analyzer cannot
  // filter out assumptions that are logically irrelevant but "touch"
  // interesting symbols; eventually it would be good to add support for this.

  // expected-note@+2 {{Assuming 'arg' is >= 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg < 0)
    return 0;
  int a = TenElements[arg];
  // expected-note@-1 {{Assuming index is less than 10, the number of 'int' elements in 'TenElements'}}
  int b = TenElements[arg + 10];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  return a + b;
}

int assumingUpperUnsigned(unsigned arg) {
  int a = TenElements[arg];
  // expected-note@-1 {{Assuming index is less than 10, the number of 'int' elements in 'TenElements'}}
  int b = TenElements[(int)arg - 10];
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset}}
  return a + b;
}

int assumingNothing(unsigned arg) {
  // expected-note@+2 {{Assuming 'arg' is < 10}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 10)
    return 0;
  int a = TenElements[arg]; // no note here, we already know that 'arg' is in bounds
  int b = TenElements[(int)arg - 10];
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset}}
  return a + b;
}

short assumingConvertedToCharP(int arg) {
  // When indices are reported, the note will use the element type that's the
  // result type of the subscript operator.
  char *cp = (char*)TenElements;
  char a = cp[arg];
  // expected-note@-1 {{Assuming index is non-negative and less than 40, the number of 'char' elements in 'TenElements'}}
  char b = cp[arg]; // no additional note, we already assumed that 'arg' is in bounds
  char c = cp[arg + 40];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 40 'char' elements}}
  return a + b + c;
}

struct foo {
  int num;
  char a[8];
  char b[5];
};

int assumingConvertedToIntP(struct foo f, int arg) {
  // When indices are reported, the note will use the element type that's the
  // result type of the subscript operator.
  int a = ((int*)(f.a))[arg];
  // expected-note@-1 {{Assuming index is non-negative and less than 2, the number of 'int' elements in 'f.a'}}
  // However, if the extent of the memory region is not divisible by the
  // element size, the checker measures the offset and extent in bytes.
  int b = ((int*)(f.b))[arg];
  // expected-note@-1 {{Assuming byte offset is less than 5, the extent of 'f.b'}}
  int c = TenElements[arg-2];
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset}}
  return a + b + c;
}

int assumingPlainOffset(struct foo f, int arg) {
  // This TC is intended to check the corner case that the checker prints the
  // shorter "offset" instead of "byte offset" when it's irrelevant that the
  // offset is measured in bytes.

  // expected-note@+2 {{Assuming 'arg' is < 2}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 2)
    return 0;

  int b = ((int*)(f.b))[arg];
  // expected-note@-1 {{Assuming byte offset is non-negative and less than 5, the extent of 'f.b'}}
  // FIXME: this should be {{Assuming offset is non-negative}}
  // but the current simplification algorithm doesn't realize that arg <= 1
  // implies that the byte offset arg*4 will be less than 5.

  int c = TenElements[arg+10];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  return b + c;
}

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t size);
void free(void *ptr);

int assumingExtent(int arg) {
  // Verify that the assumption note is printed when the extent is interesting
  // (even if the index isn't interesting).
  int *mem = (int*)malloc(arg);

  mem[12] = 123;
  // expected-note@-1 {{Assuming index '12' is less than the number of 'int' elements in the heap area}}

  free(mem);

  return TenElements[arg];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
}

int *extentInterestingness(int arg) {
  // Verify that in an out-of-bounds access issue the extent is marked as
  // interesting (so assumptions about its value are printed).
  int *mem = (int*)malloc(arg);

  TenElements[arg] = 123;
  // expected-note@-1 {{Assuming index is non-negative and less than 10, the number of 'int' elements in 'TenElements'}}

  return &mem[12];
  // expected-warning@-1 {{Out of bound access to memory after the end of the heap area}}
  // expected-note@-2 {{Access of 'int' element in the heap area at index 12}}
}
