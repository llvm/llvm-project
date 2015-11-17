__thread int tls0 = 0;
__thread int tls1 = 0;
__thread int tls2 = 1;
__thread int tls3 = 2;

int main() {
  return tls0 + tls1 + tls2;
}
