// Header for PCH test cxx1z-init-statement.cpp

constexpr int test_if(int x) { 
  if (int a = ++x; a == 0) {
    return -1;
  } else if (++a; a == 2) {
    return 0;
  }
  return 2;
}

constexpr int test_switch(int x) {
  switch (int a = ++x; a) {
    case 0:
      return -1;
    case 1:
      return 0;
    case 2:
      return 1;
  }
  return 2;
}
