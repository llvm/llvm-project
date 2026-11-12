// RUN: %check_clang_tidy %s android-cloexec-creat %t

typedef int mode_t;

extern "C" int creat(const char *path, mode_t, ...);
extern "C" int create(const char *path, mode_t, ...);

void f() {
  creat("filename", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer open() to creat() because open() allows O_CLOEXEC [android-cloexec-creat]
  // CHECK-FIXES: open ("filename", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0);
  create("filename", 0);
  // CHECK-MESSAGES-NOT: warning:
  mode_t mode = 0755;
  creat("filename", mode);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning:
  // CHECK-FIXES: open ("filename", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, mode);
}

namespace i {
int creat(const char *path, mode_t, ...);
void g() {
  creat("filename", 0);
  // CHECK-MESSAGES-NOT: warning:
}
} // namespace i

class C {
public:
  int creat(const char *path, mode_t, ...);
  void h() {
    creat("filename", 0);
    // CHECK-MESSAGES-NOT: warning:
  }
};
