int h_chain(int x) {
  return x * 2;
}

namespace chns {
int chf3(int x);

int chf2(int x) {
  return chf3(x);
}

class chcls {
public:
  int chf4(int x);
};

int chcls::chf4(int x) {
  return x * 3;
}
}
