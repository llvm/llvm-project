// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.ParameterCase: CamelCase, \
// RUN:     readability-identifier-naming.IgnoreMainLikeFunctions: true \
// RUN:  }}'

int mainLike(int argc, char **argv);
int mainLike(int argc, char **argv, const char **env);
int mainLike(int argc, const char **argv);
int mainLike(int argc, const char **argv, const char **env);
int mainLike(int argc, char *argv[]);
int mainLike(int argc, const char *argv[]);
int mainLike(int argc, char *argv[], char *env[]);
int mainLike(int argc, const char *argv[], const char *env[]);
void notMain(int argc, char **argv);
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:31: warning: invalid case style for parameter 'argv'
void notMain(int argc, char **argv, char **env);
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:31: warning: invalid case style for parameter 'argv'
// CHECK-MESSAGES: :[[@LINE-3]]:44: warning: invalid case style for parameter 'env'
int notMain(int argc, char **argv, char **env, int Extra);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: invalid case style for parameter 'argv'
// CHECK-MESSAGES: :[[@LINE-3]]:43: warning: invalid case style for parameter 'env'
int notMain(int argc, char **argv, int Extra);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: invalid case style for parameter 'argv'
int notMain(int argc, char *argv);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:29: warning: invalid case style for parameter 'argv'
int notMain(unsigned argc, char **argv);
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:35: warning: invalid case style for parameter 'argv'
int notMain(long argc, char *argv);
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:30: warning: invalid case style for parameter 'argv'
int notMain(int argc, char16_t **argv);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: invalid case style for parameter 'argv'
int notMain(int argc, char argv[]);
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:28: warning: invalid case style for parameter 'argv'
typedef char myFunChar;
typedef int myFunInt;
typedef char **myFunCharPtr;
typedef long myFunLong;
myFunInt mainLikeTypedef(myFunInt argc, myFunChar **argv);
int mainLikeTypedef(int argc, myFunCharPtr argv);
int notMainTypedef(myFunLong argc, char **argv);
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:43: warning: invalid case style for parameter 'argv'

// Don't flag as name contains the word main
int myMainFunction(int argc, char *argv[]);

// This is fine, named with wmain and has wchar ptr.
int wmainLike(int argc, wchar_t *argv[]);

// Flag this as has signature of main, but named as wmain.
int wmainLike(int argc, char *argv[]);
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for parameter 'argc'
// CHECK-MESSAGES: :[[@LINE-2]]:31: warning: invalid case style for parameter 'argv'

struct Foo {
  Foo(int argc, char *argv[]) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for parameter 'argc'
  // CHECK-MESSAGES: :[[@LINE-2]]:23: warning: invalid case style for parameter 'argv'

  int mainPub(int argc, char *argv[]);
  static int mainPubStatic(int argc, char *argv[]);

protected:
  int mainProt(int argc, char *argv[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for parameter 'argc'
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: invalid case style for parameter 'argv'
  static int mainProtStatic(int argc, char *argv[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid case style for parameter 'argc'
  // CHECK-MESSAGES: :[[@LINE-2]]:45: warning: invalid case style for parameter 'argv'

private:
  int mainPriv(int argc, char *argv[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for parameter 'argc'
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: invalid case style for parameter 'argv'
  static int mainPrivStatic(int argc, char *argv[]);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid case style for parameter 'argc'
  // CHECK-MESSAGES: :[[@LINE-2]]:45: warning: invalid case style for parameter 'argv'
};
