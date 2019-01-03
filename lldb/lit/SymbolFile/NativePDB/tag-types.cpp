// clang-format off
// REQUIRES: lld

// Test that we can display tag types.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s 
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/tag-types.lldbinit | FileCheck %s

// Test struct
struct Struct {
  // Test builtin types, which are represented by special CodeView type indices.
  bool                B;
  char                C;
  signed char         SC;
  unsigned char       UC;
  char16_t            C16;
  char32_t            C32;
  wchar_t             WC;
  short               S;
  unsigned short      US;
  int                 I;
  unsigned int        UI;
  long                L;
  unsigned long       UL;
  long long           LL;
  unsigned long long  ULL;
  float               F;
  double              D;
  long double         LD;
};

// Test class
class Class {
public:
  // Test pointers to builtin types, which are represented by different special
  // CodeView type indices.
  bool                *PB;
  char                *PC;
  signed char         *PSC;
  unsigned char       *PUC;
  char16_t            *PC16;
  char32_t            *PC32;
  wchar_t             *PWC;
  short               *PS;
  unsigned short      *PUS;
  int                 *PI;
  unsigned int        *PUI;
  long                *PL;
  unsigned long       *PUL;
  long long           *PLL;
  unsigned long long  *PULL;
  float               *PF;
  double              *PD;
  long double         *PLD;
};

// Test union
union Union {
  // Test modified types.
  const bool                *PB;
  const char                *PC;
  const signed char         *PSC;
  const unsigned char       *PUC;
  const char16_t            *PC16;
  const char32_t            *PC32;
  const wchar_t             *PWC;
  const short               *PS;
  const unsigned short      *PUS;
  const int                 *PI;
  const unsigned int        *PUI;
  const long                *PL;
  const unsigned long       *PUL;
  const long long           *PLL;
  const unsigned long long  *PULL;
  const float               *PF;
  const double              *PD;
  const long double         *PLD;
};

struct OneMember {
  int N = 0;
};


// Test single inheritance.
class Derived : public Class {
public:
  explicit Derived()
    : Reference(*this), RefMember(Member), RValueRefMember((OneMember&&)Member) {}

  // Test reference to self, to make sure we don't end up in an
  // infinite cycle.
  Derived &Reference;

  // Test aggregate class member.
  OneMember Member;

  // And modified aggregate class member.
  const OneMember ConstMember;
  volatile OneMember VolatileMember;
  const volatile OneMember CVMember;

  // And all types of pointers to class members
  OneMember *PtrMember;
  OneMember &RefMember;
  OneMember &&RValueRefMember;
};

// Test multiple inheritance, as well as protected and private inheritance.
class Derived2 : protected Class, private Struct {
public:
  // Test static data members
  static unsigned StaticDataMember;
};

unsigned Derived2::StaticDataMember = 0;

// Test scoped enums and unscoped enums.
enum class EnumInt {
  A = 1,
  B = 2
};

// Test explicit underlying types
enum EnumShort : short {
  ES_A = 2,
  ES_B = 3
};

int main(int argc, char **argv) {
  Struct S;
  Class C;
  Union U;
  Derived D;
  Derived2 D2;
  EnumInt EI;
  EnumShort ES;
  
  return 0;
}

// CHECK:      (lldb) target create "{{.*}}tag-types.cpp.tmp.exe"
// CHECK-NEXT: Current executable set to '{{.*}}tag-types.cpp.tmp.exe'
// CHECK-NEXT: (lldb) command source -s 0 '{{.*}}tag-types.lldbinit'
// CHECK-NEXT: Executing commands in '{{.*}}tag-types.lldbinit'.
// CHECK-NEXT: (lldb) type lookup -- Struct
// CHECK-NEXT: struct Struct {
// CHECK-NEXT:     bool B;
// CHECK-NEXT:     char C;
// CHECK-NEXT:     signed char SC;
// CHECK-NEXT:     unsigned char UC;
// CHECK-NEXT:     char16_t C16;
// CHECK-NEXT:     char32_t C32;
// CHECK-NEXT:     wchar_t WC;
// CHECK-NEXT:     short S;
// CHECK-NEXT:     unsigned short US;
// CHECK-NEXT:     int I;
// CHECK-NEXT:     unsigned int UI;
// CHECK-NEXT:     long L;
// CHECK-NEXT:     unsigned long UL;
// CHECK-NEXT:     long long LL;
// CHECK-NEXT:     unsigned long long ULL;
// CHECK-NEXT:     float F;
// CHECK-NEXT:     double D;
// CHECK-NEXT:     double LD;
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- Class
// CHECK-NEXT: class Class {
// CHECK-NEXT:     bool *PB;
// CHECK-NEXT:     char *PC;
// CHECK-NEXT:     signed char *PSC;
// CHECK-NEXT:     unsigned char *PUC;
// CHECK-NEXT:     char16_t *PC16;
// CHECK-NEXT:     char32_t *PC32;
// CHECK-NEXT:     wchar_t *PWC;
// CHECK-NEXT:     short *PS;
// CHECK-NEXT:     unsigned short *PUS;
// CHECK-NEXT:     int *PI;
// CHECK-NEXT:     unsigned int *PUI;
// CHECK-NEXT:     long *PL;
// CHECK-NEXT:     unsigned long *PUL;
// CHECK-NEXT:     long long *PLL;
// CHECK-NEXT:     unsigned long long *PULL;
// CHECK-NEXT:     float *PF;
// CHECK-NEXT:     double *PD;
// CHECK-NEXT:     double *PLD;
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- Union
// CHECK-NEXT: union Union {
// CHECK-NEXT:     const bool *PB;
// CHECK-NEXT:     const char *PC;
// CHECK-NEXT:     const signed char *PSC;
// CHECK-NEXT:     const unsigned char *PUC;
// CHECK-NEXT:     const char16_t *PC16;
// CHECK-NEXT:     const char32_t *PC32;
// CHECK-NEXT:     const wchar_t *PWC;
// CHECK-NEXT:     const short *PS;
// CHECK-NEXT:     const unsigned short *PUS;
// CHECK-NEXT:     const int *PI;
// CHECK-NEXT:     const unsigned int *PUI;
// CHECK-NEXT:     const long *PL;
// CHECK-NEXT:     const unsigned long *PUL;
// CHECK-NEXT:     const long long *PLL;
// CHECK-NEXT:     const unsigned long long *PULL;
// CHECK-NEXT:     const float *PF;
// CHECK-NEXT:     const double *PD;
// CHECK-NEXT:     const double *PLD;
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- Derived
// CHECK-NEXT: class Derived : public Class {
// CHECK-NEXT:     Derived &Reference;
// CHECK-NEXT:     OneMember Member;
// CHECK-NEXT:     const OneMember ConstMember;
// CHECK-NEXT:     volatile OneMember VolatileMember;
// CHECK-NEXT:     const volatile OneMember CVMember;
// CHECK-NEXT:     OneMember *PtrMember;
// CHECK-NEXT:     OneMember &RefMember;
// CHECK-NEXT:     OneMember &&RValueRefMember;
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- Derived2
// CHECK-NEXT: class Derived2 : protected Class, private Struct {
// CHECK-NEXT:     static unsigned int StaticDataMember;
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- EnumInt
// CHECK-NEXT: enum EnumInt {
// CHECK-NEXT:     A,
// CHECK-NEXT:     B
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- EnumShort
// CHECK-NEXT: enum EnumShort {
// CHECK-NEXT:     ES_A,
// CHECK-NEXT:     ES_B
// CHECK-NEXT: }
// CHECK-NEXT: (lldb) type lookup -- InvalidType
// CHECK-NEXT: no type was found matching 'InvalidType'
