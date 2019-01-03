
struct Class {
  int field;

  Class();

  Class(int x) { }

  ~Class();

  // commment
  static void method(const int &value, int defaultParam = 20);

  virtual int voidMethod(int y) const;
  void implementedMethod() const {

  }

  void outOfLineImpl(int x);

  void anotherImplementedMethod() {

  }
};
// CHECK1: "{{.*}}class.cpp" "\n\nClass::Class() { \n  <#code#>;\n}\n\nvoid Class::method(const int &value, int defaultParam) { \n  <#code#>;\n}\n" [[@LINE-1]]:3
