// The build.py script always runs the compiler in C++ mode, regardless of the
// file extension. This results in mangled names presented to the linker which
// in turn cannot find the printf symbol.
extern "C" {
int printf(const char *format, ...);

int main(int argc, char **argv) {
  printf("Hello World\n");
  return 0;
}
}
