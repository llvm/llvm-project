void foo();

void (*func)();

int main() {
  func = foo;
  func();
}
