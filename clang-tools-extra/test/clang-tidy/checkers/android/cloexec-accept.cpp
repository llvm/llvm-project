// RUN: %check_clang_tidy %s android-cloexec-accept %t

struct sockaddr {};
typedef int socklen_t;
#define NULL 0

extern "C" int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);

void f() {
  accept(0, NULL, NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: prefer accept4() to accept() because accept4() allows SOCK_CLOEXEC [android-cloexec-accept]
  // CHECK-FIXES: accept4(0, NULL, NULL, SOCK_CLOEXEC);
}

namespace i {
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
void g() {
  accept(0, NULL, NULL);
}
} // namespace i

class C {
public:
  int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
  void h() {
    accept(0, NULL, NULL);
  }
};
