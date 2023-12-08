// RUN: %check_clang_tidy %s android-cloexec-inotify-init %t

extern "C" int inotify_init();

void f() {
  inotify_init();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer inotify_init() to inotify_init1() because inotify_init1() allows IN_CLOEXEC [android-cloexec-inotify-init]
  // CHECK-FIXES: inotify_init1(IN_CLOEXEC);
}

namespace i {
int inotify_init();
void g() {
  inotify_init();
}
} // namespace i

class C {
public:
  int inotify_init();
  void h() {
    inotify_init();
  }
};
