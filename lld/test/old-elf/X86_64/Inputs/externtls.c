extern __thread  int extern_tls;

int main() {
  extern_tls = 1;
  return 0;
}
