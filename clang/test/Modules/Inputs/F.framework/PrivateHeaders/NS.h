struct NS {
  int a;
  int b;
};

enum NSE {
  FST = 22,
  SND = 43,
  TRD = 55
};

#define NS_ENUM(_type, _name) \
  enum _name : _type _name;   \
  enum _name : _type

typedef NS_ENUM(int, NSMyEnum) {
  MinX = 11,
  MinXOther = MinX,
};
