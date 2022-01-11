extern "C" int printf(const char*, ...);
extern const char* question();

inline int answer() __attribute__((always_inline));
inline int answer() { return 42; }

int main(int argc, char *argv[]) {
  int ans;
  if (argc == 1) {
    ans = 0;
  } else {
    ans = argc;
  }
  printf("%s\n", question());
  for (int i = 0; i < 10; ++i) {
    int x = answer();
    int y = answer();
    ans += x - y;
  }
  // padding to make sure question() is inlineable
  asm("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
  return ans;
}
