// Define a few different kinds of types - no modules problems.

typedef int TypeInt;

typedef TypeInt NestedTypeInt;

struct TypeStruct {
  int Member;
};

class TypeClass {
public:
  TypeClass() : Member(0) {}
private:
  int Member;
};
