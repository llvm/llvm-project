const int global_const = 42;

struct TestStruct {
  const int x;
  int y;
};

void immutable_violation_examples() {
  *(int *)&global_const = 100; // warn: Trying to write to immutable memory

  const int local_const = 42;
  *(int *)&local_const = 43; // warn: Trying to write to immutable memory

  // NOTE: The following is reported in C++, but not in C, as the analyzer
  // treats string literals as non-const char arrays in C mode.
  char *ptr_to_str_literal = (char *)"hello";
  ptr_to_str_literal[0] = 'H'; // warn: Trying to write to immutable memory

  TestStruct s = {1, 2};
  *(int *)&s.x = 10; // warn: Trying to write to immutable memory
}
