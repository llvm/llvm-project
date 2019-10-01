// RUN: %check_clang_tidy -check-suffixes=PUBLIC,NONPRIVATE,PROTECTED %s misc-non-private-member-variables-in-classes %t
// RUN: %check_clang_tidy -check-suffixes=PUBLIC,NONPRIVATE,PROTECTED %s misc-non-private-member-variables-in-classes %t -- -config='{CheckOptions: [{key: misc-non-private-member-variables-in-classes.IgnorePublicMemberVariables, value: 0}, {key: misc-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic, value: 0}]}' --
// RUN: %check_clang_tidy -check-suffixes=PUBLIC,PROTECTED %s misc-non-private-member-variables-in-classes %t -- -config='{CheckOptions: [{key: misc-non-private-member-variables-in-classes.IgnorePublicMemberVariables, value: 0}, {key: misc-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic, value: 1}]}' --
// RUN: %check_clang_tidy -check-suffixes=PUBLIC,PROTECTED %s cppcoreguidelines-non-private-member-variables-in-classes %t -- --
// RUN: %check_clang_tidy -check-suffixes=PROTECTED %s misc-non-private-member-variables-in-classes %t -- -config='{CheckOptions: [{key: misc-non-private-member-variables-in-classes.IgnorePublicMemberVariables, value: 1}, {key: misc-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic, value: 0}]}' --
// RUN: %check_clang_tidy -check-suffixes=PROTECTED %s misc-non-private-member-variables-in-classes %t -- -config='{CheckOptions: [{key: misc-non-private-member-variables-in-classes.IgnorePublicMemberVariables, value: 1}, {key: misc-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic, value: 1}]}' --

//----------------------------------------------------------------------------//

// Only data, do not warn

struct S0 {
  int S0_v0;

public:
  int S0_v1;

protected:
  int S0_v2;

private:
  int S0_v3;
};

class S1 {
  int S1_v0;

public:
  int S1_v1;

protected:
  int S1_v2;

private:
  int S1_v3;
};

// Only data and implicit or static methods, do not warn

class C {
public:
  C() {}
  ~C() {}
};

struct S1Implicit {
  C S1Implicit_v0;
};

struct S1ImplicitAndStatic {
  C S1Implicit_v0;
  static void s() {}
};

//----------------------------------------------------------------------------//

// All functions are static, do not warn.

struct S2 {
  static void S2_m0();
  int S2_v0;

public:
  static void S2_m1();
  int S2_v1;

protected:
  static void S2_m3();
  int S2_v2;

private:
  static void S2_m4();
  int S2_v3;
};

class S3 {
  static void S3_m0();
  int S3_v0;

public:
  static void S3_m1();
  int S3_v1;

protected:
  static void S3_m3();
  int S3_v2;

private:
  static void S3_m4();
  int S3_v3;
};

//============================================================================//

// union != struct/class. do not diagnose.

union U0 {
  void U0_m0();
  int U0_v0;

public:
  void U0_m1();
  int U0_v1;

protected:
  void U0_m2();
  int U0_v2;

private:
  void U0_m3();
  int U0_v3;
};

//============================================================================//

// Has non-static method with default visibility.

struct S4 {
  void S4_m0();

  int S4_v0;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S4_v0' has public visibility
public:
  int S4_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S4_v1' has public visibility
protected:
  int S4_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S4_v2' has protected visibility
private:
  int S4_v3;
};

class S5 {
  void S5_m0();

  int S5_v0;

public:
  int S5_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S5_v1' has public visibility
protected:
  int S5_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S5_v2' has protected visibility
private:
  int S5_v3;
};

//----------------------------------------------------------------------------//

// Has non-static method with public visibility.

struct S6 {
  int S6_v0;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S6_v0' has public visibility
public:
  void S6_m0();
  int S6_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S6_v1' has public visibility
protected:
  int S6_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S6_v2' has protected visibility
private:
  int S6_v3;
};

class S7 {
  int S7_v0;

public:
  void S7_m0();
  int S7_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S7_v1' has public visibility
protected:
  int S7_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S7_v2' has protected visibility
private:
  int S7_v3;
};

//----------------------------------------------------------------------------//

// Has non-static method with protected visibility.

struct S8 {
  int S8_v0;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S8_v0' has public visibility
public:
  int S8_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S8_v1' has public visibility
protected:
  void S8_m0();
  int S8_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S8_v2' has protected visibility
private:
  int S8_v3;
};

class S9 {
  int S9_v0;

public:
  int S9_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S9_v1' has public visibility
protected:
  void S9_m0();
  int S9_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S9_v2' has protected visibility
private:
  int S9_v3;
};

//----------------------------------------------------------------------------//

// Has non-static method with private visibility.

struct S10 {
  int S10_v0;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S10_v0' has public visibility
public:
  int S10_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S10_v1' has public visibility
protected:
  int S10_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S10_v2' has protected visibility
private:
  void S10_m0();
  int S10_v3;
};

class S11 {
  int S11_v0;

public:
  int S11_v1;
  // CHECK-MESSAGES-PUBLIC: :[[@LINE-1]]:7: warning: member variable 'S11_v1' has public visibility
protected:
  int S11_v2;
  // CHECK-MESSAGES-PROTECTED: :[[@LINE-1]]:7: warning: member variable 'S11_v2' has protected visibility
private:
  void S11_m0();
  int S11_v3;
};

//============================================================================//

// Static variables are ignored.
// Has non-static methods and static variables.

struct S12 {
  void S12_m0();
  static int S12_v0;

public:
  void S12_m1();
  static int S12_v1;

protected:
  void S12_m2();
  static int S12_v2;

private:
  void S12_m3();
  static int S12_v3;
};

class S13 {
  void S13_m0();
  static int S13_v0;

public:
  void S13_m1();
  static int S13_v1;

protected:
  void S13_m2();
  static int S13_v2;

private:
  void S13_m3();
  static int S13_v3;
};

struct S14 {
  void S14_m0();
  int S14_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S14_v0' has public visibility

public:
  void S14_m1();
  int S14_v1;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S14_v1' has public visibility

protected:
  void S14_m2();

private:
  void S14_m3();
};

class S15 {
  void S15_m0();

public:
  void S15_m1();
  int S15_v1;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S15_v1' has public visibility

protected:
  void S15_m2();

private:
  void S15_m3();
};

//----------------------------------------------------------------------------//

template <typename T>
struct S97 {
  void method();
  T S97_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:5: warning: member variable 'S97_v0' has public visibility
};

template struct S97<char *>;

template <>
struct S97<double> {
  void method();
  double S97d_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:10: warning: member variable 'S97d_v0' has public visibility
};

//----------------------------------------------------------------------------//

#define FIELD(x) int x;

// Do diagnose fields originating from macros.
struct S98 {
  void method();
  FIELD(S98_v0);
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:9: warning: member variable 'S98_v0' has public visibility
};

//----------------------------------------------------------------------------//

// Don't look in descendant classes.
class S99 {
  void method();

  struct S99_0 {
    int S99_S0_v0;
  };

public:
  struct S99_1 {
    int S99_S0_v0;
  };

protected:
  struct S99_2 {
    int S99_S0_v0;
  };

private:
  struct S99_3 {
    int S99_S0_v0;
  };
};

//----------------------------------------------------------------------------//

// Only diagnose once, don't let the inheritance fool you.
struct S100 {
  int S100_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S100_v0' has public visibility
  void m0();
};
struct S101_default_inheritance : S100 {
  int S101_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S101_v0' has public visibility
  void m1();
};
struct S102_public_inheritance : public S100 {
  int S102_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S102_v0' has public visibility
  void m1();
};
struct S103_protected_inheritance : protected S100 {
  int S103_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S103_v0' has public visibility
  void m1();
};
struct S104_private_inheritance : private S100 {
  int S104_v0;
  // CHECK-MESSAGES-NONPRIVATE: :[[@LINE-1]]:7: warning: member variable 'S104_v0' has public visibility
  void m1();
};
