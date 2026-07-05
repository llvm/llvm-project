int fiz() {
  return fiz();
}

int faz() {
  return faz();
}

int zip () {
  return 0;
}

int zap () {
  return 0;
}

int foo () {
  return zip();
}

int bar () {
  return zap();
}

int main() {
  return foo();
}
