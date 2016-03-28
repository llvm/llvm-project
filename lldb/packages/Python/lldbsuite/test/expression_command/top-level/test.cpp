class MyClass
{
public:
  int memberResult()
  {
    return 1;
  }
  static int staticResult()
  {
    return 1;
  }
  int externResult();
};

// --

int MyClass::externResult()
{
  return 1;
}

// --

MyClass m;

// --

enum MyEnum {
  myEnumOne = 1,
  myEnumTwo,
  myEnumThree
};

// --

class AnotherClass
{
public:
    __attribute__ ((always_inline)) int complicatedFunction() 
    {
        struct {
            int i;
        } s = { 15 };
    
        int as[4] = { 2, 3, 4, 5 };
    
        for (signed char a : as)
        {
            s.i -= a;
        }
    
        return s.i;
    }
};

// --

int doTest()
{
    return m.memberResult() + MyClass::staticResult() + m.externResult() + MyEnum::myEnumThree + myEnumOne + AnotherClass().complicatedFunction();
}

// --
