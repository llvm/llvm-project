/// This is a struct friend.
struct Foo;

// This is a nice class.
// It has some nice methods and fields.
// @brief This is a brief description.
struct MyClass {
  int PublicField;

  int myMethod(int MyParam);
  static void staticMethod();
  const int &getConst();

  enum Color { RED, GREEN, BLUE = 5 };

  typedef int MyTypedef;

  class NestedClass;

  friend struct Foo;
  /// This is a function template friend.
  template <typename T> friend void friendFunction(int);

protected:
  int protectedMethod();

  int ProtectedField;

private:
  int PrivateField;
};
