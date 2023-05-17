// RUN: %check_clang_tidy %s android-cloexec-fopen %t

#define FILE_OPEN_RO "r"

typedef int FILE;

extern "C" FILE *fopen(const char *filename, const char *mode, ...);
extern "C" FILE *open(const char *filename, const char *mode, ...);

void f() {
  fopen("filename", "r");
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'fopen' mode 'e' to set O_CLOEXEC [android-cloexec-fopen]
  // CHECK-FIXES: fopen("filename", "re");

  fopen("filename", FILE_OPEN_RO);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'fopen' mode 'e'
  // CHECK-FIXES: fopen("filename", FILE_OPEN_RO "e");

  fopen("filename", "er");
  // CHECK-MESSAGES-NOT: warning:
  fopen("filename", "re");
  // CHECK-MESSAGES-NOT: warning:
  fopen("filename", "e");
  // CHECK-MESSAGES-NOT: warning:
  open("filename", "e");
  // CHECK-MESSAGES-NOT: warning:

  char *str = "r";
  fopen("filename", str);
  // CHECK-MESSAGES-NOT: warning:
  str = "re";
  fopen("filename", str);
  // CHECK-MESSAGES-NOT: warning:
  char arr[2] = "r";
  fopen("filename", arr);
  // CHECK-MESSAGES-NOT: warning:
  char arr2[3] = "re";
  fopen("filename", arr2);
  // CHECK-MESSAGES-NOT: warning:
}

namespace i {
int *fopen(const char *filename, const char *mode, ...);
void g() {
  fopen("filename", "e");
  // CHECK-MESSAGES-NOT: warning:
}
} // namespace i

class C {
public:
  int *fopen(const char *filename, const char *mode, ...);
  void h() {
    fopen("filename", "e");
    // CHECK-MESSAGES-NOT: warning:
  }
};
