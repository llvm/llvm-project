// RUN: %clang_cc1 %s -emit-llvm-only
class T
{};

void print(const char *t);

T& operator<< (T& t,const char* c)
{
  print(c);
  return t;
}


int main()
{
  T t;
  print("foo");
  t<<"foo";
  
  return 0;
}
  
