#define DECLARE_METHODS                                                        \
  /**                                                                          \
   * @brief Declare a method to calculate the sum of two numbers               \
   */                                                                          \
  int Add(int a, int b) { return a + b; }

class MyClass {
public:
  DECLARE_METHODS
};
