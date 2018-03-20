class Patatino {
public:
  double _blah;
  Patatino(int blah) : _blah(blah) {}
};

bool operator==(const Patatino& a, const Patatino& b) {
  return a._blah < b._blah;
}
