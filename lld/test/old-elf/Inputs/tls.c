extern __thread int tls0;
extern __thread int tls1;
extern __thread int tls2;

__thread int tls0 = 0;
__thread int tls1 = 0;
__thread int tls2 = 1;

int main() {
  return tls0 + tls1 + tls2;
}
