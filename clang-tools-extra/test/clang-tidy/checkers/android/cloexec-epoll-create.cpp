// RUN: %check_clang_tidy %s android-cloexec-epoll-create %t

extern "C" int epoll_create(int size);

void f() {
  epoll_create(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer epoll_create() to epoll_create1() because epoll_create1() allows EPOLL_CLOEXEC [android-cloexec-epoll-create]
  // CHECK-FIXES: epoll_create1(EPOLL_CLOEXEC);
}

namespace i {
int epoll_create(int size);
void g() {
  epoll_create(0);
}
} // namespace i

class C {
public:
  int epoll_create(int size);
  void h() {
    epoll_create(0);
  }
};
