// This function will be embedded within the .symtab section of the
// .gnu_debugdata section.
int functionInSymtab(int num) { return num * 4; }

// This function will be embedded within the .dynsym section of the main binary.
int functionInDynsym(int num) { return num * 3; }

int main(int argc, char *argv[]) {
  int x = functionInSymtab(argc);
  int y = functionInDynsym(x);
  return y;
}
