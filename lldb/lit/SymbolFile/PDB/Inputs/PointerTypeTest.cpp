int main() {
  // Test pointer to array.
  int array[2][4];
  int(*array_pointer)[2][4] = &array;

  struct ST {
    int a;
    int f(int x) { return 1; }
  };

  ST s = {10};

  // Test pointer to a local.
  int *p_int = &s.a;

  // Test pointer to data member.
  int ST::*p_member_field = &ST::a;

  // Test pointer to member function.
  int (ST::*p_member_method)(int) = &ST::f;

  return 0;
}
