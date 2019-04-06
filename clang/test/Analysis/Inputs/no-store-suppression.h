#pragma clang system_header

namespace std {
class istream {
public:
  bool is_eof();
  char get_char();
};

istream &operator>>(istream &is, char &c) {
  if (is.is_eof())
    return;
  c = is.get_char();
}

extern istream cin;
};
