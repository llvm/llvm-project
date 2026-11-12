#define COUNT 101

static volatile int aaa;

int main(int argc, char *argv[]) {
  for (int i = 0; i < COUNT; i++)
    aaa++;
  return 0;
}
