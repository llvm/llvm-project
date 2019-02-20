
#define BUFFER_SIZE 32
struct PointType {
  int x;
  int y;
  int buffer[BUFFER_SIZE];
};

int g_global = 123;
static int s_global = 234;

int main(int argc, char const *argv[]) {
  static float s_local = 2.25;
  PointType pt = { 11,22, {0}};
  for (int i=0; i<BUFFER_SIZE; ++i)
    pt.buffer[i] = i;
  return s_global - g_global - pt.y; // breakpoint 1
}
