// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// rdar: //8620524
// PR7851
struct string {
  string (const string& );
  string ();
  ~string();
};

string operator + (char ch, const string&);

template <class T>
void IntToString(T a)
{
 string result;
 T digit; 
 char((digit < 10 ? '0' : 'a') + digit) + result;
}

int main() {
// CHECK: define linkonce_odr void @_Z11IntToStringIcEvT_(
  IntToString('a');
}

