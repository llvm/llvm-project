struct Foo { int x; };

int main(int argc, char **argv) {
  Foo f;
  f.x = 44;
  return f.x; // Set break point at this line.
}
