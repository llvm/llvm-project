// This function will be embedded within the .dynsym section of the main binary.
int functionInDynsym(int num) { return num * 3; }

int main(int argc, char *argv[]) {
  int y = functionInDynsym(argc);
  return y;
}
