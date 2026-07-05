namespace a {
extern int kGlobalInt;
extern const char *const kGlobalStr;
}

int kEvilInt = 2;

inline void f1() {
  int kGlobalInt = 3;
  const char *const kGlobalStr = "Hello2";
}
