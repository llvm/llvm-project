class Patatino {
private:
  long tinky;

public:
  Patatino(long tinky) { this->tinky = tinky; }
};

int main(void) {
  Patatino *a = new Patatino(26);
  return 0;
}
