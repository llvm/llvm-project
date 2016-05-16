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

class DiamondA
{
private:
  struct {
    int m_i;
  };
public:
  DiamondA(int i) : m_i(i) { }
  int accessor() { return m_i; }
};

// --

class DiamondB : public virtual DiamondA
{
public:
  DiamondB(int i) : DiamondA(i) { }
};

// --

class DiamondC : public virtual DiamondA
{
public:
  DiamondC(int i) : DiamondA(i) { }
};

// --

class DiamondD : public DiamondB, public DiamondC
{
public:
  DiamondD(int i) : DiamondA(i), DiamondB(i), DiamondC(i) { }
};

// --

int doTest()
{
    int a = m.memberResult();
    a += MyClass::staticResult();
    a += m.externResult();
    a += MyEnum::myEnumThree;
    a += myEnumOne;
    a += AnotherClass().complicatedFunction();
    a += DiamondD(3).accessor();
    return a;
}

// --
