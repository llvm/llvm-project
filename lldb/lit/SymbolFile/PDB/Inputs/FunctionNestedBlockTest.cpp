int main() {
  auto r = 0;
  for (auto i = 1; i <= 10; i++) {
    r += i & 1 + (i - 1) & 1 - 1;
  }

  return r;
}
